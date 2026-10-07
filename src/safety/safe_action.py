import numpy as np
import torch


class SafeActionCorrector:

    def __init__(
        self,
        cost_model,
        action_low,
        action_high,
        risk_threshold,
        num_candidates=50,
        candidate_std=0.10,
        device="cpu",
    ):
        self.cost_model     = cost_model
        self.action_low     = np.asarray(action_low, dtype=np.float32)
        self.action_high    = np.asarray(action_high, dtype=np.float32)
        self.risk_threshold = risk_threshold
        self.num_candidates = num_candidates
        self.candidate_std  = candidate_std
        self.device         = device


    def predict_risk(self, observation, actions):
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

        if observation_tensor.ndim == 1:
            observation_tensor = observation_tensor.unsqueeze(0)

        if actions_tensor.ndim == 1:
            actions_tensor = actions_tensor.unsqueeze(0)

        observation_tensor = (
            observation_tensor.expand(
                actions_tensor.shape[0],
                -1,
            )
        )

        with torch.no_grad():
            risks = self.cost_model(observation_tensor, actions_tensor)

        return risks.cpu().numpy()


    def sample_candidates(self, action):
        action = np.asarray(action, dtype=np.float32)

        noise = np.random.normal(
            loc=0.0,
            scale=self.candidate_std,
            size=(
                self.num_candidates,
                len(action),
            ),
        ).astype(np.float32)

        candidates = action[None, :] + noise

        candidates = np.clip(
            candidates,
            self.action_low,
            self.action_high,
        )

        # Include original action explicitly.
        candidates[0] = action

        return candidates


    def correct(self, observation, proposed_action):
        proposed_action = np.asarray(proposed_action, dtype=np.float32,)

        proposed_action = np.clip(
            proposed_action,
            self.action_low,
            self.action_high,
        )

        proposed_risk = self.predict_risk(
            observation,
            proposed_action,
        )[0]

        # No correction necessary.
        if proposed_risk <= self.risk_threshold:
            return {
                "action": proposed_action,
                "proposed_action": proposed_action,
                "proposed_risk": float(proposed_risk),
                "selected_risk": float(proposed_risk),
                "corrected": False,
                "safe_candidate_found": True,
                "distance": 0.0,
            }

        candidates = self.sample_candidates(proposed_action)

        risks = self.predict_risk(observation, candidates)

        safe_mask = (risks <= self.risk_threshold)

        if np.any(safe_mask):

            safe_candidates = candidates[safe_mask]
            safe_risks = risks[safe_mask]

            distances = np.linalg.norm(
                safe_candidates - proposed_action,
                axis=1,
            )

            # "Closest safe action": minimize distance to the POR action
            selected_index = np.argmin(distances)
            selected_action = safe_candidates[selected_index]
            selected_risk = safe_risks[selected_index]
            distance = distances[selected_index]

            return {
                "action": selected_action,
                "proposed_action": proposed_action,
                "proposed_risk": float(proposed_risk),
                "selected_risk": float(selected_risk),
                "corrected": True,
                "safe_candidate_found": True,
                "distance": float(distance),
            }

        # No candidate satisfies the threshold - fall back to the least risky candidate.
        selected_index = np.argmin(risks)
        selected_action = candidates[selected_index]
        selected_risk = risks[selected_index]

        distance = np.linalg.norm(selected_action - proposed_action)

        return {
            "action": selected_action,
            "proposed_action": proposed_action,
            "proposed_risk": float(proposed_risk),
            "selected_risk": float(selected_risk),
            "corrected": True,
            "safe_candidate_found": False,
            "distance": float(distance),
        }