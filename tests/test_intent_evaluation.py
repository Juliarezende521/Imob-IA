from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.evaluate import LABELS, evaluate_predictions, run_evaluation
from imob_ai.intent.report import render_evaluation_report


def test_metrics_and_confusion_matrix_cover_every_class():
    metrics = evaluate_predictions(LABELS, LABELS)
    assert metrics.accuracy == 1.0
    assert metrics.f1_macro == 1.0
    assert len(metrics.confusion_matrix) == 8
    assert all(len(row) == 8 for row in metrics.confusion_matrix)


def test_full_evaluation_uses_frozen_configuration_and_renders_report():
    result = run_evaluation(load_intent_dataset())
    assert result["selected_config"]["name"] == "word_char"
    assert sum(sum(row) for row in result["test"]["confusion_matrix"]) == 64
    report = render_evaluation_report(result)
    assert "Pipeline congelado" in report
    assert "Dummy" in report
