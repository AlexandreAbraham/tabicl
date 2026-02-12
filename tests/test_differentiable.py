from __future__ import annotations

import numpy as np
import pytest
import torch

from tabicl import TabICLClassifier


@pytest.fixture
def small_data():
    """Create a small synthetic classification dataset as tensors."""
    rng = torch.manual_seed(42)
    n_train, n_test, n_features, n_classes = 20, 5, 4, 3
    X_train = torch.randn(n_train, n_features)
    y_train = torch.randint(0, n_classes, (n_train,))
    X_test = torch.randn(n_test, n_features)
    return X_train, y_train, X_test, n_classes


@pytest.fixture
def fitted_clf(small_data):
    """Return a TabICLClassifier fitted with differentiable input."""
    X_train, y_train, _, _ = small_data
    clf = TabICLClassifier(device="cpu", differentiable_input=True)
    clf.fit_with_differentiable_input(X_train, y_train)
    return clf


class TestGradientFlow:
    def test_gradient_flows_to_x_train(self, small_data):
        """Verify gradients flow back to X_train."""
        X_train, y_train, X_test, n_classes = small_data
        X_train = X_train.clone().requires_grad_(True)

        clf = TabICLClassifier(device="cpu", differentiable_input=True)
        clf.fit_with_differentiable_input(X_train, y_train)
        logits = clf.predict_differentiable(X_test)
        loss = logits.sum()
        loss.backward()

        assert X_train.grad is not None
        assert not torch.all(X_train.grad == 0)

    def test_gradient_flows_to_x_test(self, small_data):
        """Verify gradients flow back to X_test."""
        X_train, y_train, X_test, n_classes = small_data
        X_test = X_test.clone().requires_grad_(True)

        clf = TabICLClassifier(device="cpu", differentiable_input=True)
        clf.fit_with_differentiable_input(X_train, y_train)
        logits = clf.predict_differentiable(X_test)
        loss = logits.sum()
        loss.backward()

        assert X_test.grad is not None
        assert not torch.all(X_test.grad == 0)


class TestRepeatedFit:
    def test_model_loaded_once(self, small_data):
        """Verify model is loaded only on the first fit call."""
        X_train, y_train, _, _ = small_data
        clf = TabICLClassifier(device="cpu", differentiable_input=True)

        clf.fit_with_differentiable_input(X_train, y_train)
        model_id_first = id(clf.model_)

        # Second fit with different data
        X_train2 = torch.randn_like(X_train)
        clf.fit_with_differentiable_input(X_train2, y_train)
        model_id_second = id(clf.model_)

        assert model_id_first == model_id_second

    def test_prompt_tuning_loop(self, small_data):
        """Verify loss decreases over prompt tuning iterations."""
        X_train, y_train, X_test, n_classes = small_data

        # Create a learnable prompt
        prompt = X_train.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([prompt], lr=0.1)

        clf = TabICLClassifier(device="cpu", differentiable_input=True)

        losses = []
        for _ in range(5):
            clf.fit_with_differentiable_input(prompt, y_train)
            logits = clf.predict_differentiable(X_test)
            # Use a simple target to optimize toward
            target = torch.zeros(X_test.shape[0], dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(logits, target)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should generally decrease (at least final < initial)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


class TestOutputShape:
    def test_logits_shape(self, small_data, fitted_clf):
        """Verify output shape is (n_test, n_classes)."""
        _, _, X_test, n_classes = small_data
        logits = fitted_clf.predict_differentiable(X_test)
        assert logits.shape == (X_test.shape[0], n_classes)

    def test_probabilities_shape(self, small_data, fitted_clf):
        """Verify probability output shape matches logits shape."""
        _, _, X_test, n_classes = small_data
        probs = fitted_clf.predict_differentiable(X_test, return_logits=False)
        assert probs.shape == (X_test.shape[0], n_classes)


class TestProbabilityMode:
    def test_probabilities_sum_to_one(self, small_data, fitted_clf):
        """Verify probabilities sum to 1 along class dimension."""
        _, _, X_test, _ = small_data
        probs = fitted_clf.predict_differentiable(X_test, return_logits=False)
        torch.testing.assert_close(
            probs.sum(dim=-1), torch.ones(X_test.shape[0]), atol=1e-5, rtol=1e-5
        )

    def test_probabilities_non_negative(self, small_data, fitted_clf):
        """Verify all probabilities are non-negative."""
        _, _, X_test, _ = small_data
        probs = fitted_clf.predict_differentiable(X_test, return_logits=False)
        assert (probs >= 0).all()


class TestInputValidation:
    def test_numpy_x_raises_type_error(self, small_data):
        """Verify numpy input for X raises TypeError."""
        _, y_train, _, _ = small_data
        clf = TabICLClassifier(device="cpu", differentiable_input=True)
        with pytest.raises(TypeError, match="X must be a torch.Tensor"):
            clf.fit_with_differentiable_input(np.zeros((10, 4)), y_train)

    def test_numpy_y_raises_type_error(self, small_data):
        """Verify numpy input for y raises TypeError."""
        X_train, _, _, _ = small_data
        clf = TabICLClassifier(device="cpu", differentiable_input=True)
        with pytest.raises(TypeError, match="y must be a torch.Tensor"):
            clf.fit_with_differentiable_input(X_train, np.zeros(10))

    def test_predict_before_fit_raises_runtime_error(self, small_data):
        """Verify predict_differentiable before fit raises RuntimeError."""
        _, _, X_test, _ = small_data
        clf = TabICLClassifier(device="cpu", differentiable_input=True)
        with pytest.raises(RuntimeError, match="fit_with_differentiable_input must be called"):
            clf.predict_differentiable(X_test)

    def test_too_many_classes_raises_value_error(self):
        """Verify error when n_classes exceeds model max_classes."""
        clf = TabICLClassifier(device="cpu", differentiable_input=True)
        X = torch.randn(50, 4)
        # Create y with 100 classes (> max_classes which is typically 10)
        y = torch.arange(50) % 100
        # Need enough unique classes
        y = torch.cat([torch.arange(100), torch.zeros(0, dtype=torch.long)])
        X = torch.randn(100, 4)
        with pytest.raises(ValueError, match="exceeds the max number of classes"):
            clf.fit_with_differentiable_input(X, y)
