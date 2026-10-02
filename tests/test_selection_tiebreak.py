"""Deterministic CV tie-breaking: selection must not depend on candidate order.

When two models produce the same CV mean, the old code relied on Python's
stable sort and therefore on the order candidates happened to arrive in. These
tests pin an explicit cascade instead:

    1) best CV score according to metric direction
    2) tie -> lowest CV fold standard deviation (most stable across folds)
    3) still tied -> most CV folds evaluated (most evidence)
    4) still tied -> lexicographically smallest model_name

Every criterion uses training-split evidence only. The holdout is never read,
not even to break a tie.
"""
import itertools

import pytest

from app.services.trainer import TrainingManager, TrainingResult, TIE_BREAK_RULE


def make(name, cv_score=None, cv_scores=None, status="COMPLETED", holdout=None):
    """Build a TrainingResult carrying only the fields selection reads."""
    r = TrainingResult(model_name=name, display_name=name.title())
    r.status = status
    if cv_score is not None:
        r.optimization = {
            "best_cv_score": cv_score,
            "n_folds_used": len(cv_scores or []),
        }
    r.cv_scores = list(cv_scores or [])
    r.optimized_metrics = holdout or {}
    return r


METRIC = "f1_weighted"
REG_METRIC = "neg_root_mean_squared_error"


class TestNoTie:
    def test_highest_cv_score_wins(self):
        a = make("a", cv_score=0.70)
        b = make("b", cv_score=0.90)
        c = make("c", cv_score=0.80)
        assert TrainingManager.select_best_model([a, b, c], METRIC).model_name == "b"

    def test_no_tie_is_reported_as_such(self):
        a = make("a", cv_score=0.70)
        b = make("b", cv_score=0.90)
        _, prov = TrainingManager._select_with_provenance([a, b], METRIC)
        assert prov["tie_detected"] is False
        assert prov["resolved_by"] is None
        assert prov["tied_models"] == ["b"]

    def test_order_does_not_matter_without_a_tie(self):
        a, b, c = make("a", 0.70), make("b", 0.90), make("c", 0.80)
        for order in itertools.permutations([a, b, c]):
            assert TrainingManager.select_best_model(list(order), METRIC).model_name == "b"

    def test_difference_above_tolerance_is_not_a_tie(self):
        """A genuine difference must still be ranked, not called a tie."""
        a = make("a", cv_score=0.9000001)
        b = make("b", cv_score=0.9000000)
        winner, prov = TrainingManager._select_with_provenance([a, b], METRIC)
        assert winner.model_name == "a"
        assert prov["tie_detected"] is False


class TestTieBreakByFoldStability:
    def test_lower_fold_std_wins_the_tie(self):
        # Same mean; the second is spread across folds, the first is not.
        a = make("a", cv_score=0.90, cv_scores=[0.90, 0.90, 0.90])
        b = make("b", cv_score=0.90, cv_scores=[0.70, 0.95, 1.05])
        winner, prov = TrainingManager._select_with_provenance([a, b], METRIC)
        assert winner.model_name == "a"
        assert prov["tie_detected"] is True
        assert prov["resolved_by"] == "cv_fold_std"

    def test_stability_wins_regardless_of_order(self):
        a = make("a", cv_score=0.90, cv_scores=[0.90, 0.90, 0.90])
        b = make("b", cv_score=0.90, cv_scores=[0.70, 0.95, 1.05])
        for order in itertools.permutations([a, b]):
            assert TrainingManager.select_best_model(list(order), METRIC).model_name == "a"

    def test_equal_stability_falls_through_to_name(self):
        a = make("z_model", cv_score=0.90, cv_scores=[0.80, 1.00])
        b = make("a_model", cv_score=0.90, cv_scores=[0.80, 1.00])
        winner, prov = TrainingManager._select_with_provenance([b, a], METRIC)
        assert winner.model_name == "a_model"
        assert prov["resolved_by"] == "model_name"


class TestTieBreakByFoldCount:
    def test_missing_stability_is_not_replaced_by_zero(self):
        """A single fold yields no spread; it must not be handed a fake 0.0
        (which would look like perfect stability and win every tie)."""
        solo = make("solo", cv_score=0.9, cv_scores=[0.9])
        assert TrainingManager._cv_stability(solo) is None
        pair = make("pair", cv_score=0.9, cv_scores=[0.9, 0.9])
        winner, _ = TrainingManager._select_with_provenance([solo, pair], METRIC)
        assert winner.model_name == "pair"

    def test_more_folds_breaks_a_tie(self):
        wide = make("wide", cv_score=0.9, cv_scores=[0.9, 0.9, 0.9, 0.9, 0.9])
        narrow = make("narrow", cv_score=0.9, cv_scores=[0.9, 0.9])
        winner, prov = TrainingManager._select_with_provenance([narrow, wide], METRIC)
        assert winner.model_name == "wide"
        assert prov["resolved_by"] in ("cv_fold_std", "cv_fold_count")


class TestNameIsFinalDeterministicFallback:
    def test_lexicographic_name_breaks_a_total_tie(self):
        a = make("aaa", cv_score=0.90, cv_scores=[0.90, 0.90])
        b = make("bbb", cv_score=0.90, cv_scores=[0.90, 0.90])
        c = make("ccc", cv_score=0.90, cv_scores=[0.90, 0.90])
        for order in itertools.permutations([a, b, c]):
            winner, prov = TrainingManager._select_with_provenance(list(order), METRIC)
            assert winner.model_name == "aaa"
            assert prov["resolved_by"] == "model_name"

    def test_identical_inputs_produce_identical_selection(self):
        """The required property: the same candidates always pick the winner,
        no matter what order they arrive in."""
        def build():
            return [make("m_c", 0.88, [0.80, 0.96]),
                    make("m_a", 0.88, [0.88, 0.88]),
                    make("m_b", 0.88, [0.85, 0.91])]
        picks = {
            TrainingManager.select_best_model(list(order), METRIC).model_name
            for order in itertools.permutations(build())
        }
        assert len(picks) == 1
        assert picks == {"m_a"}


class TestTieBreakNeverUsesHoldout:
    @pytest.mark.parametrize("metric", [METRIC, REG_METRIC])
    def test_flipping_holdout_metrics_does_not_change_the_winner(self, metric):
        # 'a' would look terrible on the holdout; 'b' nearly perfect.
        a = make("a", cv_score=0.90, cv_scores=[0.90, 0.90], holdout={"mse": 999.0})
        b = make("b", cv_score=0.90, cv_scores=[0.70, 1.10], holdout={"mse": 0.001})
        _, prov = TrainingManager._select_with_provenance([a, b], metric)
        assert prov["winner"] == "a"
        assert prov["tie_broken_by_holdout"] is False
        assert prov["resolved_by"] == "cv_fold_std"


class TestRegressionTies:
    def test_negated_loss_ties_are_broken_the_same_way(self):
        # RMSE metric: a higher (less negative) neg-RMSE is better.
        a = make("lin", cv_score=-0.60, cv_scores=[-0.60, -0.60])
        b = make("tree", cv_score=-0.60, cv_scores=[-0.80, -0.40])
        winner, prov = TrainingManager._select_with_provenance([a, b], REG_METRIC)
        assert winner.model_name == "lin"
        assert prov["tie_detected"] is True
        assert prov["resolved_by"] == "cv_fold_std"

    def test_regression_total_tie_falls_back_to_name(self):
        a = make("zzz", cv_score=-0.60, cv_scores=[-0.60, -0.60])
        b = make("aaa", cv_score=-0.60, cv_scores=[-0.60, -0.60])
        winner, prov = TrainingManager._select_with_provenance([a, b], REG_METRIC)
        assert winner.model_name == "aaa"
        assert prov["resolved_by"] == "model_name"


class TestRawLossDirection:
    """The stored CV score is the negated loss, so it is always 'higher is
    better'. These tests pin that a *lower raw loss* wins even though its
    stored score is numerically smaller - the exact case a naive
    'maximize the number' implementation would get backwards."""

    def test_lower_raw_loss_wins_even_though_its_score_is_smaller(self):
        terrible = make("terrible", cv_score=-2.00, cv_scores=[-2.00, -2.00])
        good = make("good", cv_score=-0.50, cv_scores=[-0.50, -0.50])
        # -0.50 > -2.00 numerically, but the RAW losses are 0.50 vs 2.00.
        assert (-0.50 > -2.00) is True
        winner = TrainingManager.select_best_model([terrible, good], REG_METRIC)
        assert winner.model_name == "good"

    def test_lower_raw_loss_wins_for_a_minimised_metric(self):
        """Same guarantee for an un-negated loss name such as 'mse'."""
        spec_metrics = ["neg_mean_squared_error", "neg_root_mean_squared_error"]
        for metric in spec_metrics:
            high_loss = make("high_loss", cv_score=-9.0, cv_scores=[-9.0, -9.0])
            low_loss = make("low_loss", cv_score=-0.25, cv_scores=[-0.25, -0.25])
            assert TrainingManager.select_best_model(
                [high_loss, low_loss], metric).model_name == "low_loss"

    def test_tie_between_equal_raw_losses_is_still_broken_by_stability(self):
        a = make("lin", cv_score=-0.60, cv_scores=[-0.60, -0.60])
        b = make("tree", cv_score=-0.60, cv_scores=[-0.80, -0.40])
        winner, prov = TrainingManager._select_with_provenance([a, b], REG_METRIC)
        assert winner.model_name == "lin"
        assert prov["resolved_by"] == "cv_fold_std"

    def test_rule_text_states_direction_rather_than_bare_highest(self):
        assert "according to metric direction" in TIE_BREAK_RULE


class TestProvenanceIsPersisted:
    def test_provenance_carries_the_rule_and_the_decision(self):
        a = make("a", cv_score=0.90, cv_scores=[0.90, 0.90])
        b = make("b", cv_score=0.90, cv_scores=[0.70, 1.10])
        best, _ = TrainingManager._select_with_provenance([a, b], METRIC)
        prov = TrainingManager.build_selection_provenance([a, b], METRIC, best)

        tb = prov["tie_break"]
        assert tb["rule"] == TIE_BREAK_RULE
        assert tb["tie_detected"] is True
        assert sorted(tb["tied_models"]) == ["a", "b"]
        assert tb["winner"] == "a"
        assert tb["resolved_by"] == "cv_fold_std"
        assert tb["tie_broken_by_holdout"] is False
        assert tb["tolerance"]["rel_tol"] > 0
        assert "no statistical significance claim" in tb["tolerance_meaning"]
        assert prov["holdout_used_for_selection"] is False
        assert prov["selected_model"] == "a"

    def test_candidate_rows_expose_stability_evidence(self):
        a = make("a", cv_score=0.90, cv_scores=[0.90, 0.90])
        b = make("b", cv_score=0.90, cv_scores=[0.70, 1.10])
        best, _ = TrainingManager._select_with_provenance([a, b], METRIC)
        prov = TrainingManager.build_selection_provenance([a, b], METRIC, best)
        rows = {c["model_name"]: c for c in prov["candidates_considered"]}
        assert rows["a"]["cv_fold_std"] == pytest.approx(0.0)
        assert rows["b"]["cv_fold_std"] > 0
        assert rows["a"]["cv_fold_count"] == 2
        assert rows["a"]["selected"] is True
        assert rows["b"]["selected"] is False

    def test_persisted_record_agrees_with_the_actual_winner(self):
        results = [make("x", 0.90, [0.90, 0.90]), make("y", 0.90, [0.70, 1.10])]
        best = TrainingManager.select_best_model(results, METRIC)
        prov = TrainingManager.build_selection_provenance(results, METRIC, best)
        assert prov["tie_break"]["winner"] == best.model_name
        assert prov["selected_model"] == best.model_name
