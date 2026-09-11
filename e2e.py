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

            loaded_model = mlflow.transformers.load_model(
                model_uri
            )

            assert loaded_model is not None

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
    # 8. Read everything back from the registered model
    # ---------------------------------------------------------

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

        reloaded_model = mlflow.transformers.load_model(model_uri)

        assert reloaded_model is not None

        texts = [summarization_input]

        result = reloaded_model(texts)

        assert result is not None
        assert len(result) == len(texts)
