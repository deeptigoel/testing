"""
End-to-end integration test for the summarization -> registration ->
evaluation -> promotion -> model-card pipeline.

This test intentionally reuses the same fixtures and variable names as:
  - test_summarization.py   (mlflow_server, mlflow_client, summarization_input,
                              local_summarization_model, run_id, models,
                              summary_results)
  - test_mlflow_promotion.py (parent_run_id, child_run_id, eval_results,
                               eval_df_composite, promotion_info,
                               champion/challenger tags & metrics)

and mirrors every logging/tagging call made by mlflow_pipeline_ipynb.py:
    set_run_tags, log_environment, log_parameters, set_source_lineage_tags,
    mlflow.data.from_pandas + log_input, log_artifact(s), update_run_tags,
    log_transformer_model, register_model, log_model_card,
    get_incumbent_champion_score, calculate_composite_score,
    prepare_promotion_metadata, log_metrics, build_model_card,
    save_model_card, tag_model_version_with_card.

NOTES ON DEVIATIONS FROM THE PIPELINE SCRIPT (flagged, not silently fixed):

1. set_run_tags() in the pipeline is called with description=/product=/
   run_scope=/run_id= kwargs that don't exist on the real
   mlflow_utils.set_run_tags(run_type, capability, dataset_version, git_sha).
   That call would raise TypeError as written in the notebook - here we call
   it with the signature that actually exists in mlflow_utils.py.

2. build_model_card / save_model_card / log_model_card /
   tag_model_version_with_card aren't defined in any uploaded file. They're
   imported here from `model_card_utils`, matching the mlflow_utils /
   promotion_utils naming convention - update the import if they live
   elsewhere. `log_model_card` is also used for the pre-evaluation static
   card step, matching how the pipeline calls it with a single path there
   vs. a list of paths later.

3. validate_model() and log_model_artifact() aren't defined anywhere either.
   validate_model's result is stood in for with validation_status=True
   (tagged accordingly); log_model_artifact(MODEL_OUTPUT_DIR) is skipped
   since there's no equivalent local directory in this test's scope.

4. PRODUCT_NAME / chunk_size / batch_size / max_length / min_tokens aren't
   in any provided config.py, so test-local values are used, taken directly
   from summarize.py's own defaults (chunk_text's max_tokens_for_chunking,
   recursive_reduce's batch_size, summarize()'s max_new_tokens/min_length).

5. get_incumbent_champion_score() is defined locally inside
   mlflow_pipeline_ipynb.py (not importable from a module), so it's
   reproduced verbatim below rather than imported.

6. summarizer.temp_load_models is NOT monkeypatched - summarize.py defines
   and calls its own local temp_load_models(), so the patch target is
   summarize.temp_load_models (see prior conversation - test_summarization.py
   has the same bug patching the wrong module).
"""

import time

import mlflow
import pandas as pd
import pytest
from mlflow.models import infer_signature, set_signature, get_model_info

import mlflow_utils
import summarize
from promotion_utils import calculate_composite_score, prepare_promotion_metadata
from model_card_utils import (
    build_model_card,
    save_model_card,
    log_model_card,
    tag_model_version_with_card,
)


def get_incumbent_champion_score(client, registered_names):
    """
    Reproduced verbatim from mlflow_pipeline_ipynb.py, since it's defined
    locally there rather than in an importable module.
    """
    best = None

    for registered_name in set(registered_names):
        for version in client.search_model_versions(
            f"name='{registered_name}'"
        ):
            if version.tags.get("promotion_role") != "champion":
                continue

            raw_score = version.tags.get("composite_score")

            if raw_score is None:
                continue

            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue

            if best is None or score > best:
                best = score

    return best


@pytest.mark.integration
def test_end_to_end_register_evaluate_promote_model_card(
    mlflow_server,
    mlflow_client,
    summarization_input,
    local_summarization_model,
    monkeypatch,
    tmp_path,
):
    """
    Verifies the full lifecycle:

    summarization -> model logging -> model registration ->
    evaluation (nested run) -> promotion decision ->
    model-version tagging -> model card creation ->
    read-back from the registered model.
    """

    start_time = time.time()

    # ---------------------------------------------------------
    # 1. Configure MLflow for the integration-test environment
    # ---------------------------------------------------------

    monkeypatch.setattr(
        mlflow_utils,
        "USE_UNITY_CATALOG",
        False,
    )

    monkeypatch.setattr(
        mlflow_utils,
        "MLFLOW_REGISTRY_URI",
        mlflow_server,
    )

    monkeypatch.setattr(
        mlflow_utils,
        "MLFLOW_TRACKING_URI",
        mlflow_server,
    )

    mlflow_utils.configure_mlflow()

    # ---------------------------------------------------------
    # 2. Mock model loading with two aliases of the same real
    #    local model, so promotion has a champion AND a challenger
    #    to compare (mirrors test_mlflow_promotion.py's model_1/model_2)
    # ---------------------------------------------------------

    monkeypatch.setattr(
        summarize,
        "temp_load_models",
        lambda: {
            "distilbart_cnn_6_6": local_summarization_model,
            "distilbart_cnn_6_6_challenger": local_summarization_model,
        },
    )

    # ---------------------------------------------------------
    # 3. Run actual summarization pipeline
    # ---------------------------------------------------------

    summary_results, models = summarize.run_pipeline(
        summarization_input
    )

    assert summary_results
    assert models

    for summary in summary_results.values():
        assert isinstance(summary, str)
        assert summary.strip()

    dataset_version = "test-dataset-v1"

    # Test-local stand-ins for config.py values not in any uploaded file,
    # taken from summarize.py's own defaults.
    PRODUCT_NAME = "test-product"
    CHUNK_SIZE = 600     # chunk_text()'s max_tokens_for_chunking default
    BATCH_SIZE = 3        # recursive_reduce()'s batch_size default
    MAX_LENGTH = 400      # summarize()'s final max_new_tokens
    MIN_TOKENS = 40        # summarize()'s min_length floor

    # ---------------------------------------------------------
    # 3a. Prepare artifacts, mirroring mlflow_pipeline_ipynb.py:
    #       - df logged as an MLflow *dataset* (mlflow.data.from_pandas +
    #         mlflow.log_input), not just a plain text file
    #       - the raw source file also logged as a file artifact
    #       - summary output merges df + summary_results into one
    #         file, logged at the artifact root (no artifact_path),
    #         mirroring save_summary(df, summary_results, output_file)
    #       - a synthetic groundtruth folder (mirrors
    #         ground_truth/<product>/*.xlsx from evaluation.py)
    #       - a placeholder "existing" model card file, for the
    #         pre-evaluation log_model_card(MODEL_CARD_FILE) step
    # ---------------------------------------------------------

    input_text_path = tmp_path / "input.txt"
    input_text_path.write_text(summarization_input)

    df = pd.DataFrame({"text": [summarization_input]})

    input_dataset = mlflow.data.from_pandas(
        df,
        source=str(input_text_path),
        name="summarization_input",
    )

    output_df = df.copy()

    for model_name, summary in summary_results.items():
        output_df[f"summary_{model_name}"] = summary

    summary_output_path = tmp_path / "summary_output.csv"
    output_df.to_csv(summary_output_path, index=False)

    groundtruth_dir = tmp_path / "groundtruth"
    groundtruth_dir.mkdir()
    (groundtruth_dir / "reference.txt").write_text(summarization_input)

    existing_model_card_path = tmp_path / "existing_model_card.md"
    existing_model_card_path.write_text(
        "# Model Card (placeholder, pre-evaluation)\n"
    )

    # ---------------------------------------------------------
    # 4. Parent MLflow run: log, validate, and register every model
    # ---------------------------------------------------------

    registered_models = []

    with mlflow.start_run(
        run_name="summarization-integration-test"
    ) as run:

        run_id = run.info.run_id
        parent_run_id = run_id

        # -----------------------------------------
        # 4a. Run tags, environment, parameters, source lineage
        # (mlflow_utils.set_run_tags' real signature only takes
        # run_type/capability/dataset_version/git_sha)
        # -----------------------------------------

        mlflow_utils.set_run_tags(
            run_type="inference",
            capability="abstractive summarization",
            dataset_version=dataset_version,
            git_sha="test-git-sha",
        )

        mlflow_utils.log_environment()

        mlflow_utils.log_parameters(
            {
                "chunk_size": CHUNK_SIZE,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "min_tokens": MIN_TOKENS,
            }
        )

        mlflow_utils.set_source_lineage_tags(
            source_file_type="txt",
            source_path=str(input_text_path),
        )

        mlflow.log_param(
            "input_source",
            "sqlite",
        )

        mlflow.log_metric(
            "prediction_count",
            len(summary_results),
        )

        # -----------------------------------------
        # 4b. Input dataset lineage + raw source file
        # -----------------------------------------

        mlflow.log_input(
            input_dataset,
            context="inference",
        )

        mlflow_utils.log_artifact(
            str(input_text_path),
            artifact_path="input",
        )

        mlflow_utils.update_run_tags(
            model_name=",".join(models.keys()),
            number_of_model=str(len(models)),
        )

        # -----------------------------------------
        # 4c. Summary output (df + summary_results merged), logged
        # at the artifact root - matches
        # save_summary(df, summary_results, output_file) + log_artifact(output_file)
        # -----------------------------------------

        mlflow_utils.log_artifact(
            str(summary_output_path),
        )

        # -----------------------------------------
        # 4d. Pre-evaluation "existing model card" step
        # -----------------------------------------

        log_model_card(str(existing_model_card_path))

        for model_name, model_pipeline in models.items():

            # -----------------------------------------
            # 4e. Log
            # -----------------------------------------

            assert mlflow_utils.log_transformer_model(
                model_name=model_name,
                model_pipeline=model_pipeline,
            )

            model_uri = (
                f"runs:/{run_id}/{model_name}"
            )

            # summarization_input (from conftest) is the input example
            # this model was actually run against - infer + attach its
            # signature so it's captured on the logged model artifact.
            model_input_example = pd.DataFrame(
                {"text": [summarization_input]}
            )

            raw_prediction = model_pipeline(summarization_input)

            model_output_example = pd.DataFrame(
                {"prediction": [raw_prediction[0]["summary_text"]]}
            )

            signature = infer_signature(
                model_input_example,
                model_output_example,
            )

            set_signature(model_uri, signature)

            validation_status = True  # validate_model() isn't provided

            mlflow_utils.update_run_tags(
                validation_status=(
                    "passed" if validation_status else "failed"
                ),
            )

            # -----------------------------------------
            # 4f. Register
            # -----------------------------------------

            registered_info = mlflow_utils.register_model(
                model_uri=model_uri,
                model_name=model_name,
                description="Abstractive summarization model",
                run_id=parent_run_id,
                dataset_version=dataset_version,
                validation_status=validation_status,
                client=mlflow_client,
            )

            assert registered_info["registered_name"]
            assert registered_info["version"]

            mlflow_utils.update_run_tags(
                registration_status="registered",
            )

            registered_models.append(
                {
                    "model_name": model_name,
                    "registered_name": registered_info["registered_name"],
                    "version": registered_info["version"],
                }
            )

        assert registered_models

        # -------------------------------------------------------
        # 5. Evaluation child run
        # -------------------------------------------------------

        with mlflow.start_run(
            run_name="evaluation",
            nested=True,
        ) as evaluation_run:

            child_run_id = evaluation_run.info.run_id

            # Verify parent-child relationship
            assert (
                evaluation_run.data.tags[mlflow.parent_run_id]
                == parent_run_id
            )

            mlflow_utils.set_run_tags(
                run_type="evaluation",
                capability="abstractive summarization evaluation",
                dataset_version=dataset_version,
                git_sha="test-git-sha",
            )

            mlflow_utils.update_run_tags(
                evaluation_run_id=child_run_id,
            )

            # -----------------------------------------
            # 5a. Groundtruth used for evaluation
            # -----------------------------------------

            mlflow_utils.log_artifacts(
                str(groundtruth_dir),
                artifact_path="groundtruth",
            )

            model_names = [m["model_name"] for m in registered_models]

            # Evaluation results (synthetic scores, same shape as
            # test_mlflow_promotion.py's eval_results)
            eval_results = pd.DataFrame(
                {
                    "model_name": model_names,
                    "ROUGE": [0.85, 0.65][: len(model_names)],
                    "METEOR": [0.80, 0.60][: len(model_names)],
                }
            )

            # Calculate composite score
            eval_df_composite = calculate_composite_score(eval_results)

            assert not eval_df_composite.empty
            assert "composite" in eval_df_composite.columns

            # -----------------------------------------
            # 5b. Evaluation results artifact
            # -----------------------------------------

            eval_output_path = tmp_path / "evaluation_results.csv"
            eval_df_composite.to_csv(eval_output_path, index=False)

            mlflow_utils.log_artifact(
                str(eval_output_path),
                artifact_path="eval_output",
            )

            # -----------------------------------------
            # 5c. Composite-score row -> numeric metrics + string tags,
            # matching the metrics/params split around
            # log_metrics(metrics) + log_eval_quality_tags(params)
            # -----------------------------------------

            composite_row = eval_df_composite.iloc[0].to_dict()

            eval_run_metrics = {}
            eval_quality_params = {}

            for key, value in composite_row.items():
                if isinstance(value, (int, float)):
                    eval_run_metrics[key] = float(value)
                else:
                    eval_quality_params[key] = str(value)

            if eval_run_metrics:
                mlflow_utils.log_metrics(eval_run_metrics)

            if eval_quality_params:
                mlflow_utils.update_run_tags(**eval_quality_params)

            mlflow_utils.update_run_tags(evaluation_status="complete")

            # Prepare promotion metadata - incumbent champion score is
            # computed for real (not hardcoded None); on this first run
            # it resolves to None since no version is tagged
            # promotion_role="champion" yet.
            incumbent_score = get_incumbent_champion_score(
                mlflow_client,
                [m["registered_name"] for m in registered_models],
            )

            promotion_info = prepare_promotion_metadata(
                eval_df_composite,
                incumbent_champion_score=incumbent_score,
            )

            assert promotion_info["promotion_status"] == "Completed"
            assert promotion_info["champion_model_name"] in model_names
            assert promotion_info["challenger_model_name"] in model_names

            # Log promotion metadata to evaluation run
            mlflow_utils.update_run_tags(
                champion_model=promotion_info["champion_model_name"],
                challenger_model=promotion_info["challenger_model_name"],
                promotion_status=promotion_info["promotion_status"],
            )

            mlflow.log_metric(
                "champion_composite_score",
                promotion_info["champion_score"],
            )
            mlflow.log_metric(
                "challenger_composite_score",
                promotion_info["challenger_score"],
            )

            # -------------------------------------------------
            # 6. Link evaluation results back to each registered
            #    model version (tags + composite score + role)
            # -------------------------------------------------

            for registered_model in registered_models:

                model_name = registered_model["model_name"]
                registered_name = registered_model["registered_name"]
                version = registered_model["version"]

                model_row = eval_df_composite[
                    eval_df_composite["model_name"] == model_name
                ].iloc[0]

                composite_score = float(model_row["composite"])

                mlflow_client.set_model_version_tag(
                    name=registered_name,
                    version=version,
                    key="evaluation_run_id",
                    value=child_run_id,
                )

                promotion_role = promotion_info["aliases"].get(model_name)

                if promotion_role:
                    mlflow_client.set_model_version_tag(
                        name=registered_name,
                        version=version,
                        key="promotion_role",
                        value=promotion_role,
                    )

                mlflow_client.set_model_version_tag(
                    name=registered_name,
                    version=version,
                    key="composite_score",
                    value=str(composite_score),
                )

                # -------------------------------------------
                # 7. Build + save + log the model card
                # -------------------------------------------

                eval_metrics = {}
                eval_params = {}

                for key, value in model_row.to_dict().items():
                    try:
                        eval_metrics[key] = float(value)
                    except (ValueError, TypeError):
                        eval_params[key] = str(value)

                model_card = build_model_card(
                    model_name=model_name,
                    registered_name=registered_name,
                    model_version=version,
                    parent_run_id=parent_run_id,
                    evaluation_run_id=child_run_id,
                    validation_status="passed",
                    dataset_version=dataset_version,
                    evaluation_metrics=eval_metrics,
                    evaluation_params=eval_params,
                    promotion_role=promotion_role,
                    composite_score=composite_score,
                    promotion_status=promotion_info["promotion_status"],
                    promotion_blocked_reason=promotion_info.get(
                        "promotion_blocked_reason"
                    ),
                )

                assert model_card

                card_output_dir = (
                    tmp_path / "model_cards" / f"{model_name}_{version}"
                )

                card_paths = save_model_card(
                    model_card,
                    output_dir=str(card_output_dir),
                )

                assert card_paths

                log_model_card(card_paths)

                tag_model_version_with_card(
                    client=mlflow_client,
                    registered_name=registered_name,
                    version=version,
                    model_card_path="model_card/model_card.json",
                )

            # Verify promotion metadata is NOT present on parent run
            retrieved_parent = mlflow_client.get_run(parent_run_id)
            parent_run_tags = retrieved_parent.data.tags

            assert "champion_model" not in parent_run_tags
            assert "challenger_model" not in parent_run_tags
            assert "promotion_status" not in parent_run_tags

        # ---------------------------------------------------------
        # 8. Final pipeline status
        # ---------------------------------------------------------

        execution_time = time.time() - start_time

        mlflow_utils.log_metrics(
            {"execution_time_seconds": execution_time}
        )

        mlflow_utils.update_run_tags(pipeline_status="completed")

    # ---------------------------------------------------------
    # 9. Read everything back: run tags/artifacts + registered model
    # ---------------------------------------------------------

    retrieved_parent_run = mlflow_client.get_run(parent_run_id)
    parent_tags = retrieved_parent_run.data.tags

    assert parent_tags["run_type"] == "inference"
    assert parent_tags["pipeline_status"] == "completed"
    assert parent_tags["registration_status"] == "registered"
    assert parent_tags["validation_status"] == "passed"

    parent_artifact_paths = {
        artifact.path
        for artifact in mlflow_client.list_artifacts(parent_run_id)
    }

    assert "input" in parent_artifact_paths
    assert "summary_output.csv" in parent_artifact_paths

    logged_dataset_names = {
        dataset_input.dataset.name
        for dataset_input in retrieved_parent_run.inputs.dataset_inputs
    }

    assert "summarization_input" in logged_dataset_names

    retrieved_evaluation_run = mlflow_client.get_run(child_run_id)
    evaluation_tags = retrieved_evaluation_run.data.tags

    assert evaluation_tags["run_type"] == "evaluation"
    assert evaluation_tags["evaluation_status"] == "complete"
    assert evaluation_tags["evaluation_run_id"] == child_run_id

    child_artifact_paths = {
        artifact.path
        for artifact in mlflow_client.list_artifacts(child_run_id)
    }

    assert "groundtruth" in child_artifact_paths
    assert "eval_output" in child_artifact_paths

    for registered_model in registered_models:

        model_name = registered_model["model_name"]
        registered_name = registered_model["registered_name"]
        version = registered_model["version"]

        retrieved_version = mlflow_client.get_model_version(
            name=registered_name,
            version=version,
        )

        assert retrieved_version.tags["evaluation_run_id"] == child_run_id
        assert "composite_score" in retrieved_version.tags
        assert retrieved_version.tags.get(
            "model_card_path"
        ) == "model_card/model_card.json"

        model_uri = f"models:/{registered_name}/{version}"

        model_info = get_model_info(model_uri)
        assert model_info.signature is not None

        reloaded_model = mlflow.transformers.load_model(model_uri)

        assert reloaded_model is not None

        texts = [summarization_input]

        result = reloaded_model(texts)

        assert result is not None
        assert len(result) == len(texts)




######
## Summary

This PR includes the following updates to improve test coverage for the MLflow summarization workflow:

### 1. Resolved issues from the previous PR

* Addressed the 4 issues identified during the review of the previous PR.
* Incorporated the required changes and updated the relevant test/code accordingly.

### 2. Added unit test cases

* Added/updated unit test cases for the relevant utility and workflow functions.
* Covered different input and validation scenarios to ensure the individual functions behave as expected.

### 3. Added integration test cases

* Added 4 integration test cases covering different scenarios across the MLflow workflow.
* Instead of having one large integration test covering the complete flow, the scenarios have been split into separate test cases to provide better coverage and make individual failures easier to identify and troubleshoot.
* The integration tests currently cover different stages/scenarios of the MLflow lifecycle, including promotion/evaluation and model registration/artifact logging.

### 4. System test coverage

* A separate end-to-end system test will cover the complete workflow as a single flow.
* This will provide full system-level validation while keeping the integration tests focused on individual scenarios.

## Test Strategy

The current approach is:

**Unit Tests → Individual functions/components**
**Integration Tests → 4 scenario-based tests covering different MLflow interactions**
**System Test → 1 complete end-to-end workflow**

This separation is intended to provide focused failure identification at the integration level while still maintaining complete end-to-end coverage through the system test.

## Validation

* All added/updated pytest test cases have been executed successfully.
* Existing test cases were also validated to ensure there are no regressions.

Lightweight Test Model Setup
Created a small script to generate/load a lightweight model for integration testing, similar to the Hugging Face model flow used by the application.
This provides a lightweight and controlled model setup for local integration testing without depending on the Databricks workspace/Unity Catalog model.
For system testing, the actual model will be retrieved from the Databricks workspace/Unity Catalog once the required environment connectivity/setup is available.


ntegration coverage has been split into four scenario-based test cases, with a separate end-to-end system test planned to validate the complete workflow.”
  The integration tests cover the identified scenarios individually, while the complete end-to-end system test will validate the overall workflow. Any scenarios or interactions not covered by the integration tests will be additionally validated during system testing to ensure comprehensive coverage.




  ### Testing Experimentation

As part of the integration testing experimentation, two MLflow tracking fixtures were added to `conftest.py`:

1. **Real MLflow server + SQLite backend** — used to validate integration with an actual MLflow tracking server.
2. **Direct SQLite tracking** — used for selected scenarios without starting an MLflow server.

These setups were explored to understand the MLflow tracking behaviour and determine the appropriate approach for different test scenarios. The server-based setup was used where applicable, while one test currently uses the direct SQLite approach due to local server stability issues encountered during testing.

This is considered a pragmatic setup for the current integration tests. The **real MLflow server-based workflow will be covered and validated through the complete end-to-end system test**.


### Additional Fixes and Validations

* **Promotion threshold logic:** Validated the promotion threshold logic and confirmed that the promotion criteria/card are aligned with the expected threshold behaviour.

* **Dependency/version conflicts:** Resolved the existing `Hugging Face Hub`, `mlflow`, and metrics/transformers version compatibility issues. Since moving to a different server/environment was not required at this stage, the conflicting dependency (`mlflow`-related scoring component) was removed and the scoring implementation was aligned with the existing `transformers` and metrics versions.

* **Model version registration:** Validated model version registration to ensure that the registered model/version is created successfully and is correctly visible in the MLflow UI/Experiments as expected.

* **Score precision:** Verified that scores are not rounded prematurely before the relevant calculations/promotion logic are applied, ensuring that the original score precision is retained throughout the calculation flow.
