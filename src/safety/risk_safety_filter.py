import numpy as np
import torch


class RiskSafetyFilter:
    """
    Probability-based safety filter for POR actions.

    The filter implements the proposed Safe-POR procedure:

    1. POR proposes an action a_POR.
    2. Evaluate:

           P(violation within K steps | s, a_POR)

    3. If the predicted risk is <= risk_threshold, execute a_POR unchanged.

    4. Otherwise, sample candidate actions within an L2 radius rho around
       a_POR.

    5. Evaluate all candidates with the risk model.

    6. Among candidates satisfying:

           risk(s, a) <= risk_threshold

       choose the action closest to a_POR.

    7. If no candidate satisfies the threshold, execute the candidate with
       the lowest predicted risk.

    This is a soft safety mechanism. It does not mathematically guarantee
    constraint satisfaction because safety depends on the learned risk model.

    Parameters
    ----------
    risk_model : torch.nn.Module
        Trained RiskPredictor.

    action_low : array-like
        Lower bounds of the continuous action space.

    action_high : array-like
        Upper bounds of the continuous action space.

    risk_threshold : float
        Maximum acceptable predicted probability of a safety violation.

        For example:

            risk_threshold = 0.20

        means actions are accepted when the model predicts at most a 20%
        probability of a violation within the K-step horizon.

    num_candidates : int, optional
        Number of candidate actions evaluated when intervention is required.

        Candidate 0 is always the original POR action.

    search_radius : float, optional
        Maximum L2 distance rho between a sampled candidate and the
        original POR action.

    device : str or torch.device, optional
        Device used for risk-model inference.

    rng : np.random.Generator, optional
        Random number generator used for candidate sampling.

        A seeded generator can be supplied for reproducible evaluation:

            np.random.default_rng(42)
    """

    def __init__(
        self,
        risk_model,
        action_low,
        action_high,
        risk_threshold,
        num_candidates=50,
        search_radius=0.10,
        device="cpu",
        rng=None,
    ):
        self.risk_model = risk_model

        self.action_low = np.asarray(
            action_low,
            dtype=np.float32,
        ).reshape(-1)

        self.action_high = np.asarray(
            action_high,
            dtype=np.float32,
        ).reshape(-1)

        self.risk_threshold = float(
            risk_threshold
        )

        self.num_candidates = int(
            num_candidates
        )

        self.search_radius = float(
            search_radius
        )

        self.device = torch.device(device)

        # Use NumPy's Generator API rather than the global random state.
        self.rng = (
            rng
            if rng is not None
            else np.random.default_rng()
        )

        # -----------------------------------------------------------
        # Validate configuration.
        # -----------------------------------------------------------

        if self.action_low.shape != self.action_high.shape:
            raise ValueError(
                "action_low and action_high must have the same shape"
            )

        if np.any(
            self.action_low > self.action_high
        ):
            raise ValueError(
                "every action_low value must be <= action_high"
            )

        if not (
            0.0
            <= self.risk_threshold
            <= 1.0
        ):
            raise ValueError(
                "risk_threshold must be between 0 and 1"
            )

        if self.num_candidates < 1:
            raise ValueError(
                "num_candidates must be at least 1"
            )

        if self.search_radius < 0.0:
            raise ValueError(
                "search_radius must be non-negative"
            )

        # The safety filter only uses the network for inference.
        self.risk_model.to(self.device)
        self.risk_model.eval()

    def predict_risk(
        self,
        observation,
        actions,
    ):
        """
        Predict violation probabilities for one or more candidate actions.

        A single state is repeated across the action batch. This allows all
        candidate actions to be evaluated in one neural-network forward pass.

        Parameters
        ----------
        observation : array-like
            Current normalized state.

        actions : array-like
            Either one action with shape (action_dim,) or multiple actions
            with shape (num_actions, action_dim).

        Returns
        -------
        np.ndarray
            Predicted violation probabilities.
        """

        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )

        actions_tensor = torch.as_tensor(
            actions,
            dtype=torch.float32,
            device=self.device,
        )

        # Convert a single state into a batch of size 1.
        if observation_tensor.ndim == 1:
            observation_tensor = (
                observation_tensor.unsqueeze(0)
            )

        # Convert a single action into a batch of size 1.
        if actions_tensor.ndim == 1:
            actions_tensor = (
                actions_tensor.unsqueeze(0)
            )

        # Normally only one current state is supplied. Repeat that state
        # for every candidate action so they can all be scored at once.
        if observation_tensor.shape[0] == 1:

            observation_tensor = (
                observation_tensor.expand(
                    actions_tensor.shape[0],
                    -1,
                )
            )

        elif (
            observation_tensor.shape[0]
            != actions_tensor.shape[0]
        ):
            raise ValueError(
                "observation batch size must be 1 or "
                "match the action batch size"
            )

        with torch.no_grad():

            probabilities = (
                self.risk_model.predict_proba(
                    observation_tensor,
                    actions_tensor,
                )
            )

        return (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

    def sample_candidates(self, proposed_action):
        noise = np.random.normal(
            loc=0.0,
            scale=self.search_radius,
            size=(self.num_candidates, proposed_action.shape[0]),
        )

        candidates = proposed_action + noise

        candidates = np.clip(
            candidates,
            self.action_low,
            self.action_high,
        )

        return candidates

    def correct(
        self,
        observation,
        proposed_action,
    ):
        """
        Apply the safety filter to an action proposed by POR.

        Parameters
        ----------
        observation : array-like
            Current normalized observation.

        proposed_action : array-like
            Action proposed by POR.

        Returns
        -------
        dict
            Dictionary containing:

            action
                Action that should actually be executed.

            proposed_action
                Original action proposed by POR.

            proposed_risk
                Predicted K-step violation probability for the POR action.

            selected_risk
                Predicted risk of the action actually selected.

            corrected
                Whether the safety layer intervened.

            safe_candidate_found
                Whether an action satisfying the risk threshold was found.

            distance
                L2 distance between the POR action and selected action.
        """

        proposed_action = np.asarray(
            proposed_action,
            dtype=np.float32,
        ).reshape(-1)

        proposed_action = np.clip(
            proposed_action,
            self.action_low,
            self.action_high,
        )

        # -----------------------------------------------------------
        # Step 1:
        # Evaluate the action originally proposed by POR.
        # -----------------------------------------------------------

        proposed_risk = float(
            self.predict_risk(
                observation,
                proposed_action,
            )[0]
        )

        # -----------------------------------------------------------
        # Step 2:
        # If POR's action already satisfies the safety threshold,
        # do not interfere with the policy.
        # -----------------------------------------------------------

        if (
            proposed_risk
            <= self.risk_threshold
        ):
            return {
                "action": proposed_action,
                "proposed_action": proposed_action,
                "proposed_risk": proposed_risk,
                "selected_risk": proposed_risk,
                "corrected": False,
                "safe_candidate_found": True,
                "distance": 0.0,
            }

        # -----------------------------------------------------------
        # Step 3:
        # POR's proposed action is considered unsafe.
        # Generate alternatives in the local action-space neighbourhood.
        # -----------------------------------------------------------

        candidates = self.sample_candidates(
            proposed_action
        )

        # Evaluate all candidates in one batched neural-network pass.
        risks = self.predict_risk(
            observation,
            candidates,
        )

        # -----------------------------------------------------------
        # Step 4:
        # Find all actions satisfying the risk constraint:
        #
        #     P(violation | s, a) <= epsilon
        # -----------------------------------------------------------

        safe_mask = (
            risks <= self.risk_threshold
        )

        if np.any(safe_mask):

            safe_candidates = (
                candidates[safe_mask]
            )

            safe_risks = (
                risks[safe_mask]
            )

            # Compute the distance from every safe candidate to the
            # action POR originally wanted to execute.
            distances = np.linalg.norm(
                safe_candidates
                - proposed_action,
                axis=1,
            )

            # -------------------------------------------------------
            # Select the closest safe action:
            #
            # argmin ||a - a_POR||_2
            #
            # subject to:
            #
            # risk(s, a) <= epsilon
            # -------------------------------------------------------

            selected_index = int(
                np.argmin(distances)
            )

            selected_action = (
                safe_candidates[
                    selected_index
                ]
            )

            selected_risk = float(
                safe_risks[
                    selected_index
                ]
            )

            distance = float(
                distances[
                    selected_index
                ]
            )

            return {
                "action": selected_action,
                "proposed_action": proposed_action,
                "proposed_risk": proposed_risk,
                "selected_risk": selected_risk,
                "corrected": True,
                "safe_candidate_found": True,
                "distance": distance,
            }

        # -----------------------------------------------------------
        # Step 5:
        # No sampled candidate satisfies the threshold.
        #
        # Fall back to the candidate with the smallest predicted
        # probability of a violation.
        # -----------------------------------------------------------

        selected_index = int(
            np.argmin(risks)
        )

        selected_action = (
            candidates[selected_index]
        )

        selected_risk = float(
            risks[selected_index]
        )

        distance = float(
            np.linalg.norm(
                selected_action
                - proposed_action
            )
        )

        return {
            "action": selected_action,
            "proposed_action": proposed_action,
            "proposed_risk": proposed_risk,
            "selected_risk": selected_risk,
            "corrected": True,
            "safe_candidate_found": False,
            "distance": distance,
        }