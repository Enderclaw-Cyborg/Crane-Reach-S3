"""A working Crane starter agent.

Each unit runs a separate instance of this class. This starter walks forward until it sees an
enemy, then takes one legal step toward the nearest visible enemy and names it. Start at the
``TODO(you)`` comments.
Read ``environment.md`` beside this file for the rules, helpers, and first improvement. Prepare
episode state in ``reset``. The constructor takes no arguments.
"""

from sandbox.crane import action, me, paths, tile, units, visible
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation


class Agent:
    """Marches toward the enemy side, then steps toward the nearest visible enemy."""

    def reset(self, seed, observation) -> None:
        # Called once before each match. The opening observation is available here for
        # precomputation outside the decision clock. This starter stores no state.
        pass

    def _abilities_enabled(self, observation: SkirmishObservation) -> bool:
        """Return whether the active season has enabled cavalry charges and shield walls."""
        return bool(observation["observation"]["parameters"].get("unit_abilities", 0))

    def _can_charge(self, observation: SkirmishObservation, start: AxialPosition, path_id: int) -> bool:
        """Return whether a legal path ends far enough away to trigger the cavalry charge bonus."""
        if not self._abilities_enabled(observation) or path_id == 0:
            return False
        landing = tile.at_path_end(start, path_id)
        return tile.distance(start, landing) >= 3

    def _shield_wall_ready(self, observation: SkirmishObservation, here: AxialPosition, ally: AxialPosition) -> bool:
        """Return whether a footman standing here would be adjacent to an allied footman."""
        if not self._abilities_enabled(observation):
            return False
        return tile.distance(here, ally) == 1

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        # The enemies this unit can see.
        enemies = visible.enemies(observation)

        if not enemies:
            # At the beginning of a default skirmish match, units sit apart and see no enemies.
            # me.direction is the digit toward the enemy side, so this unit heads that way.
            here = me.position(observation)
            forward = me.direction(observation)
            legal_steps = action.legal_steps(observation)

            # Prefer the forward step when it is legal and not obviously worse terrain. When the
            # direct route is blocked or bad ground, choose the easiest legal tile instead.
            if forward in legal_steps:
                forward_tile = tile.at_path_end(here, forward)
                forward_penalty = self._terrain_penalty(observation, forward_tile)
                if not any(
                    self._terrain_penalty(observation, tile.at_path_end(here, step)) < forward_penalty
                    for step in legal_steps
                    if step != forward
                ):
                    return action.move(forward)

            if legal_steps:
                best_step = min(
                    legal_steps,
                    key=lambda step: (
                        self._terrain_penalty(observation, tile.at_path_end(here, step)),
                        0 if step == forward else 1,
                    ),
                )
                return action.move(best_step)

            # TODO(you): this unit stands still when something blocks the way.
            # It may still attack, but can you choose a better response?
            return action.stay()

        # This unit's current {"q": ..., "r": ...} position.
        here = me.position(observation)

        # The closest enemy in sight. min returns the enemy dictionary, not the distance.
        nearest = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))

        unit_type = me.unit_type(observation)
        if unit_type == "archer":
            step = self._step_archer(observation, nearest["position"])
        elif unit_type == "cavalry":
            step = self._step_cavalry(observation, nearest["position"])
        elif unit_type == "footman":
            step = self._step_footman(observation, nearest["position"])
        else:
            step = self._step_toward(observation, nearest["position"])

        # Naming a target makes the strike prefer that enemy. Any visible enemy can be named,
        # so both orders below are legal.
        if step == 0:
            return action.stay(nearest["unit_id"], observation)
        return action.move(step, nearest["unit_id"], observation)

    def _terrain_penalty(self, observation: SkirmishObservation, position: AxialPosition) -> int:
        """Return a rough movement penalty for standing on this tile."""
        standing = tile.terrain_at(observation, position)
        terrain = standing["terrain"]
        feature = standing["feature"]

        if terrain in {"water", "void"}:
            return 99

        penalty = {"grass": 0, "hill": 2}.get(terrain, 1)
        penalty += {"none": 0, "forest": 1, "marsh": 2, "waste": 2}.get(feature, 0)
        return penalty

    def _step_toward(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Return a legal path that closes the gap to goal, or 0 when none does."""
        here = me.position(observation)

        # Standing still is path id 0. A step must reduce the distance to be worth taking,
        # but when two options are equally close we prefer the calmer ground.
        best_step = 0
        if me.unit_type(observation) == "cavalry":
            best_score = (
                tile.distance(here, goal),
                0,
                self._terrain_penalty(observation, here),
            )
        else:
            best_score = (
                tile.distance(here, goal) + self._terrain_penalty(observation, here),
                self._terrain_penalty(observation, here),
                tile.distance(here, goal),
            )
        detour = 0
        detour_score = None

        for path_id in action.legal_paths(observation):
            landing = tile.at_path_end(here, path_id)
            step_distance = tile.distance(landing, goal)
            path_length = len(paths.decode(path_id))
            if path_id:
                candidate_detour_score = (
                    step_distance,
                    -path_length if me.unit_type(observation) == "cavalry" else path_length,
                    self._terrain_penalty(observation, landing),
                )
                if detour_score is None or candidate_detour_score < detour_score:
                    detour, detour_score = path_id, candidate_detour_score

            if me.unit_type(observation) == "cavalry":
                path_score = (
                    step_distance,
                    -path_length,
                    self._terrain_penalty(observation, landing),
                )
            else:
                path_score = (
                    step_distance + self._terrain_penalty(observation, landing),
                    self._terrain_penalty(observation, landing),
                    step_distance,
                )

            # Remember this path if it is the best score so far.
            if path_score < best_score:
                best_step, best_score = path_id, path_score

        # A water seam may require moving away from the enemy before reaching a passage.
        return best_step if best_step else detour

    def _step_footman(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Prefer adjacent ally coverage when the shield wall is active, otherwise follow the usual approach."""
        here = me.position(observation)
        if self._abilities_enabled(observation):
            shield_paths: list[tuple[int, int]] = []
            for path_id in action.legal_paths(observation):
                if not path_id:
                    continue
                landing = tile.at_path_end(here, path_id)
                if any(
                    self._shield_wall_ready(observation, landing, ally["position"])
                    and ally["type"] == "footman"
                    for ally in visible.allies(observation)
                ):
                    shield_paths.append((tile.distance(landing, goal), path_id))
            if shield_paths:
                return min(shield_paths)[1]
        return self._step_toward(observation, goal)

    def _step_cavalry(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Choose a legal path that reaches a charge, and otherwise keep pressing the enemy."""
        here = me.position(observation)
        if self._abilities_enabled(observation):
            charge_paths: list[tuple[tuple[int, int, int], int]] = []
            for path_id in action.legal_paths(observation):
                if not path_id:
                    continue
                landing = tile.at_path_end(here, path_id)
                if self._can_charge(observation, here, path_id):
                    charge_paths.append(
                        (
                            (
                                tile.distance(landing, goal),
                                -len(paths.decode(path_id)),
                                self._terrain_penalty(observation, landing),
                            ),
                            path_id,
                        )
                    )
            if charge_paths:
                return min(charge_paths)[1]
        return self._step_toward(observation, goal)

    def _step_archer(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Retreat as far as possible while keeping the enemy within attack range."""
        here = me.position(observation)
        attack_range = units.STATS["archer"].attack_range
        current_distance = tile.distance(here, goal)
        candidates = []

        for path_id in action.legal_paths(observation):
            landing = tile.at_path_end(here, path_id)
            distance = tile.distance(landing, goal)
            if distance <= attack_range:
                candidates.append(
                    (
                        distance,
                        -len(paths.decode(path_id)),
                        self._terrain_penalty(observation, landing),
                        path_id,
                    )
                )

        if candidates:
            return max(candidates)[-1]

        # If the enemy is still outside the archer's range, close the gap first.
        if current_distance > attack_range:
            return self._step_toward(observation, goal)
        return 0

    # Optional: a reinforcement-learning hook called after every step with that step's
    # transition. Its time counts against the timing and episode budget. The order argument is
    # what act returned. It is named order so it does not shadow the action helpers.
    #
    # def learn(self, observation, order: SkirmishAction, reward: float, terminated: bool) -> None:
    #     ...

    # Optional: messaging. Season settings enable it from Season 3 onward. When enabled, chat runs
    # after a unit chooses its order and receives messages that arrived since its previous
    # activation. Return each message with a recipient and text. Use None to broadcast to both
    # sides, or a player id such as "player_2", not a unit id, to send directly to one ally. The
    # rosters in the observation map each player to its unit. By default, text is limited to 200
    # characters.
    # A direct message reaches its allied unit at its next activation, after that unit chooses its
    # own order. Every message is recorded and shown in replays, so nothing you send is ever secret.
    # Return nothing to stay silent.
    #
    # def chat(self, inbox: list[dict]) -> list[dict] | None:
    #     ...
