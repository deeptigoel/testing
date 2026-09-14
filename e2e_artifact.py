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

It walks the full pipeline end to end:
    load model -> log model -> register model -> evaluate ->
    decide promotion (champion/challenger) -> tag model versions ->
    build + log model card -> read everything back from the
    registered model.

NOTE:
    build_model_card / save_model_card / log_model_card /
    tag_model_version_with_card are referenced by mlflow_pipeline_ipynb.py
    but their source module wasn't provided alongside the other files.
    They're imported here from `model_card_utils`, matching the naming
    convention of `mlflow_utils` / `promotion_utils`. Update the import
    below if they actually live elsewhere in your project.
"""

import mlflow
import pandas as pd
import pytest
from mlflow.models import infer_signature, set_signature, get_model_info

import mlflow_utils
import summarizer
import summarize
from promotion_utils import calculate_composite_score, prepare_promotion_metadata
from model_card_utils import (
    build_model_card,
    save_model_card,
    log_model_card,
    tag_model_version_with_card,
)


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
        summarizer,
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

    # ---------------------------------------------------------
    # 4. Parent MLflow run: log, validate, and register every model
    # ---------------------------------------------------------

    registered_models = []

    with mlflow.start_run(
        run_name="summarization-integration-test"
    ) as run:

        run_id = run.info.run_id
        parent_run_id = run_id

        mlflow.log_param(
            "input_source",
            "sqlite",
        )

        mlflow.log_metric(
            "prediction_count",
            len(summary_results),
        )

        # -----------------------------------------
        # 4a-artifacts. Input dataset lineage + raw
        # source file (matches mlflow.data.from_pandas
        # + mlflow.log_input + log_artifact(..., "input"))
        # -----------------------------------------

        mlflow.log_input(
            input_dataset,
            context="inference",
        )

        mlflow_utils.log_artifact(
            str(input_text_path),
            artifact_path="input",
        )

        # NOTE: update_run_tags() isn't in any uploaded file, so these
        # are set directly via mlflow.set_tag to reproduce the same
        # model_name / number_of_model tags the pipeline sets.
        mlflow.set_tag(
            "model_name",
            ",".join(models.keys()),
        )
        mlflow.set_tag(
            "number_of_model",
            str(len(models)),
        )

        # -----------------------------------------
        # 4a-artifacts. Summary output (df + summary_results
        # merged), logged at the artifact root - no artifact_path,
        # matching save_summary(df, summary_results, output_file)
        # -----------------------------------------

        mlflow_utils.log_artifact(
            str(summary_output_path),
        )

        for model_name, model_pipeline in models.items():

            # -----------------------------------------
            # 4a. Log
            # -----------------------------------------

            assert mlflow_utils.log_transformer_model(
                model_name=model_name,
                model_pipeline=model_pipeline,
            )

            model_uri = (
                f"runs:/{run_id}/{model_name}"
            )

            # summarization_input (from conftest) is the input example
            # this model was actually run against — infer + attach its
            # signature so it's captured on the logged model artifact.
            # This mirrors the infer_signature(model_input, model_output)
            # block commented out in test_summarization.py, which notes
            # log_transformer_model() doesn't currently accept a
            # signature argument, so it's attached post-log instead.

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

            # -----------------------------------------
            # 4b. Register
            # -----------------------------------------

            registered_info = mlflow_utils.register_model(
                model_uri=model_uri,
                model_name=model_name,
                description="Abstractive summarization model",
                run_id=parent_run_id,
                dataset_version=dataset_version,
                validation_status=True,
                client=mlflow_client,
            )

            assert registered_info["registered_name"]
            assert registered_info["version"]

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

            # -----------------------------------------
            # 5a-artifacts. Groundtruth used for evaluation
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
            # 5b-artifacts. Evaluation results
            # -----------------------------------------

            eval_output_path = tmp_path / "evaluation_results.csv"
            eval_df_composite.to_csv(eval_output_path, index=False)

            mlflow_utils.log_artifact(
                str(eval_output_path),
                artifact_path="eval_output",
            )

            # NOTE: log_eval_quality_tags() / update_run_tags() aren't in
            # any uploaded file. Reproducing their effect directly: any
            # non-numeric composite-score column becomes a run tag, and
            # we mark the evaluation run's status like the pipeline does.
            eval_quality_params = {
                key: str(value)
                for key, value in eval_df_composite.iloc[0].to_dict().items()
                if not isinstance(value, (int, float))
            }

            if eval_quality_params:
                mlflow.set_tags(eval_quality_params)

            mlflow.set_tag("evaluation_status", "complete")

            # Prepare promotion metadata
            promotion_info = prepare_promotion_metadata(
                eval_df_composite,
                incumbent_champion_score=None,
            )

            assert promotion_info["promotion_status"] == "Completed"
            assert promotion_info["champion_model_name"] in model_names
            assert promotion_info["challenger_model_name"] in model_names

            # Log promotion metadata to evaluation run
            mlflow.set_tag(
                "champion_model",
                promotion_info["champion_model_name"],
            )
            mlflow.set_tag(
                "challenger_model",
                promotion_info["challenger_model_name"],
            )
            mlflow.set_tag(
                "promotion_status",
                promotion_info["promotion_status"],
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
    # 8. Read everything back: run artifacts + registered model
    # ---------------------------------------------------------

    parent_artifact_paths = {
        artifact.path
        for artifact in mlflow_client.list_artifacts(parent_run_id)
    }

    assert "input" in parent_artifact_paths
    assert "summary_output.csv" in parent_artifact_paths

    retrieved_parent_run = mlflow_client.get_run(parent_run_id)
    logged_dataset_names = {
        dataset_input.dataset.name
        for dataset_input in retrieved_parent_run.inputs.dataset_inputs
    }

    assert "summarization_input" in logged_dataset_names

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
