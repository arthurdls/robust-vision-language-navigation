
"""
Notes:
On Uncertainty Quantification:
    Papers:
        ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees: https://arxiv.org/abs/2407.00499
            Accepted by EMNLP 2024 Findings
        API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access: https://arxiv.org/abs/2403.01216
            Accepted by EMNLP 2024 Findings
    Pipeline (share this paper)
        Step 1: Sample Generation: For a single prompt, the LLM is queried multiple times (e.g., 20-30 times) with high temperature (randomness) to produce a diverse set of responses.
        Step 2: Semantic Clustering: The pipeline groups the text responses based on their meaning, not just their exact wording. For example, "The capital is Paris" and "Paris is the capital" would go into the same semantic cluster.
        Step 3: Uncertainty Scoring (Nonconformity): A "nonconformity score" (where high score = high uncertainty) is calculated for each response. This score is based on black-box heuristics like:
            Frequency: How often does this semantic meaning appear in the samples?
            Similarity: How semantically similar is this response to the other generated responses?
            Diversity: How many different semantic clusters were generated in total?
        Step 4: Calibration (Conformal Prediction): The nonconformity scores from a separate calibration dataset are used to find a specific threshold (called the $\hat{q}$ quantile). This threshold is calculated to meet a user-defined correctness guarantee, such as 90%.
        Step 5: Output a "Prediction Set": For a new prompt, the model generates its set of responses (from Step 1). The system filters these responses, keeping only those with an uncertainty score below the threshold from Step. The final output is this "prediction set," which is statistically guaranteed to contain the correct answer 90% of the time.

Tasks to complete:
    Work on Vision Model
        Navigation constraints & Object understanding
        Vision model may be able to augment predicate
        Take into account different embodiments (drone, car, manipulator)
        Vision interacts with language model for navigation
        Start: Drone (with capabilities explained)
        - Give examples (navigation, presence of required objects)
        - LLM-Vision feedback loop on pipeline
    Tests:
        Moving in directions
        Manipulating objects in scene (and feasability of manipulations)
            Missing object to manipulate, things it cannot do, things it can do etc
        Add a few more situations for feasability model
"""

import json
from .utils.base_llm_providers import LLMFactory
from .utils.other_utils import extract_json


class Camera:
    def __init__(self, frames: list[str]):
        self.frames = frames
        self.frame_idx = 0

    def get_next_frame(self):
        if self.frame_idx >= len(self.frames):
            return None
        else:
            next_frame = self.frames[self.frame_idx]
            self.frame_idx += 1
            return next_frame


class Feasibility_Checker:
    """
    This class is a light-weight fast evaluator. Its only job is to
    critically evaluate a *specific* action against an image and
    return a structured JSON response.
    """
    # Next Steps (Add uncertainty quantification):
    #   Instead of querying the Checker once, we should query it 5-10 times
    #   with high temperature. If 80% say "Feasible", we proceed.
    #   If 50/50, we treat it as "Infeasible" due to high uncertainty.

    def __init__(self, model="gpt-4o-mini"):
        print(f"Initializing Feasibility_Checker with model: {model}")
        self._llm = LLMFactory.create(model=model, rate_limit_seconds=0.0)
        self.high_level_task = None

    def set_high_level_task(self, high_level_task):
        self.high_level_task = high_level_task

    def check_action_feasibility(self, image: str, action_to_check: str) -> dict:
        """
        Checks if a single low-level action is feasible given the current image.

        Args:
            image: The URL or path to the image.
            action_to_check: The specific low-level action (e.g., "move forward 10 feet").

        Returns:
            dict: A dictionary with "feasible" (bool) and "reasoning" (str).
        """
        if not self.high_level_task:
            raise Exception("[Planner Error] Set high level task before requesting next action")
        # prompt = f"""
        # You are a robot's Safety and Feasibility Checker. Your job is critical.
        # You must evaluate a proposed action against an image for feasibility.

        # ### Robot Constraints:
        # * **Body:** You are a 2-foot wide quadcopter.
        # * **Gaps:** Any gap (doorway, space between chairs) MUST be > 3 feet to be "Navigable".
        # * **Gripper:** You have a 1-foot gripper reach. To "pick up" an object,
        #     it must be clearly visible and estimated to be within 1 foot.
        # * **Collision:** You CANNOT move through ANY solid objects (walls, people,
        #     furniture, closed doors).

        # ### Task:
        # Analyze the image and the proposed action.

        # **Proposed Action:** "{action_to_check}"

        # **Checks to perform:**
        # 1.  **Obstacles:** Is the immediate path for this action blocked? (EXCEPTION: If action is "wait", ignore obstacles. "Wait" is valid even if path is blocked).
        # 2.  **Navigation:** Is there enough space (e.g., > 3-foot gap) for the drone's body?
        # 3.  **Manipulation:** If the action is some variation of "pick up [object]", is that object
        #     present AND within the 1-foot reach?
        # 4.  **Hazards:** Are there any other obvious hazards (e.g., a person walking,
        #     a closed glass door)?

        # ### Output:
        # Respond ONLY with a valid JSON object in the following format.
        # Do not add any other text.

        # ```json
        # {{
        #   "feasible": <true | false>,
        #   "reasoning": "<Your concise reason for the verdict and suggested alternative based on image contents if infeasible. e.g., 'Path is clear.'
        #     or 'Infeasible: Path blocked by a chair. I suggest you go above the chair or around the chair.'
        #     or 'Infeasible: 'cup' is not visible. I suggest you look around for the cup.'
        #     or 'Infeasible: 'cup' is visible but > 1 foot away. I suggest you get closer to the cup.'>"
        # }}
        # ```
        # """

        prompt = f"""
        You are a robot's Safety and Feasibility Checker.
        You must evaluate a proposed action against an image for feasibility.

        ### Robot Constraints:
        * **Body:** 2-foot wide quadcopter.
        * **Gaps:** > 3 feet to be "Navigable".
        * **Reach:** within 1 foot (for "pick up").
        * **Collision:** Cannot move THROUGH solid objects

        ### High-Level Goal: "{self.high_level_task}"

        ### Task:
        Analyze the image and the **Proposed Action**: "{action_to_check}"

        ### Checks to perform (TRAJECTORY ANALYSIS):

        1.  **Forward / Approach:**
            * **Trajectory:** Moves into the center of the image.
            * **Check:** Is the path blocked?
            * **CRITICAL EXCEPTION (The "Approach" Rule):** If the "obstacle" directly in front is a **Surface** (desk, table, counter) and the robot's goal is to interact with an object ON that surface (e.g., "pick up cup"), moving forward/approaching IS FEASIBLE.
              * **Wall/Person in front?** -> INFEASIBLE (Collision).
              * **Desk/Table in front (with target object)?** -> FEASIBLE (Interaction).
            * **...UNLESS (The "Stop" Rule):** The target object/surface is **ALREADY within reach** (e.g., < 1 foot away, fills the camera view).
              * If already close: Moving forward is **INFEASIBLE** (Collision Risk).
              * **Example Suggestion:** if higher level goal has 'pick up ball' you might suggest "Object is already in reach. Do not move forward. I suggest you 'pick up ball'."

        2.  **Backward:**
            * **Check:** Generally feasible unless backed into a corner.

        3.  **Ascend (Go Up):**
            * **Trajectory:** Moves vertically UP.
            * **Rule:** Objects on the ground or in front (desks, chairs) DO NOT block "ascend".

        4.  **Descend (Go Down):**
            * **Rule:** Do not descend if already landed or hovering inches above a surface.

        5.  **Strafe (Left / Right):**
            * **Rule:** Objects directly *in front* do not block a strafe. Only objects to the *side* block strafe.

        6.  **Rotate (Yaw):**
            * **Rule:** A wall or obstacle directly in front DOES NOT block rotation.

        7.  **Manipulation (Pick Up):**
            * **Check:** Is the target object visible AND estimated within 1 foot?

        8.  **Wait:**
            * **Rule:** Always FEASIBLE.

        ### Output:
        Respond ONLY with a valid JSON object:
        ```json
        {{
          "feasible": <true | false>,
          "reasoning": "<Explain based on the logic above. If INFEASIBLE, provide a constructive suggestion that helps the High Level Goal.
             BAD Suggestion: 'Path blocked by desk, turn around.' (fails goal)
             GOOD Suggestion: 'Path blocked by desk, but since cup is there, try 'pick up cup' or 'ascend' to hover over it.'>"
        }}
        ```
        """

        try:
            # API call
            response_string = self._llm.make_text_and_image_request(prompt, image)
            result = extract_json(response_string)
            return result

        except json.JSONDecodeError:
            print(f"[Checker Error] Failed to decode JSON from: {response_string}")
            return {"feasible": False, "reasoning": "Feasibility checker returned invalid JSON."}
        except Exception as e:
            print(f"[Checker Error] An unexpected error occurred: {e}")
            return {"feasible": False, "reasoning": f"Checker exception: {e}"}


class LLM_Planner:
    """
    This is the central processing unit for the robot. It holds the high-level goal
    and proposes low-level actions. It uses feedback from the
    Feasibility_Checker to correct itself.
    """
    def __init__(self, model="gpt-4o-mini"):
        print(f"Initializing LLM_Planner with model: {model}")
        self._llm = LLMFactory.create(model=model, rate_limit_seconds=0.0)
        self.action_history = []
        self.proposed_action_history = [] # This should be cleared after each action
        # TODO: Scaffold for progress metric
        self.progress_metric = 0.0
        self.high_level_task = None

    def set_high_level_task(self, high_level_task):
        self.high_level_task = high_level_task

    def propose_next_action(self, image, feasibility_feedback=None):
        """
        Proposes the next single low-level action.

        Args:
            image (str): The URL or path to the image.
            feasibility_feedback (dict, optional): The JSON output from the
                                                 Feasibility_Checker if the
                                                 last attempt failed.

        Returns:
            str: The proposed low-level action (e.g., "turn right 90 degrees").
        """
        if not self.high_level_task:
            raise Exception("[Planner Error] Set high level task before requesting next action")

        # Build the dynamic part of the prompt based on feedback
        feedback_prompt_segment = ""
        if feasibility_feedback and not feasibility_feedback.get('feasible', True):
            feedback_prompt_segment = f"""
            ### CRITICAL FEEDBACK
            Your last action was **Infeasible**.
            **Reason:** "{feasibility_feedback.get('reasoning', 'No reason given.')}"
            You MUST propose a *different* action to overcome this obstacle.
            Do not propose the same action again.
            """
            # Exception may need to be added for dynamic scene (i.e. different environment, camera image)

        # Maybe we also add a 'look around' action to turn the camera 360 degrees and identify objects/openings
        # to inform future actions
        prompt = f"""
        You are the AI Planner for an autonomous drone with a manipulator.
        Your job is to propose a single, small, low-level action to make
        progress toward a high-level goal.

        ### Robot Embodiment:
        * **Body:** 2-foot wide quadcopter (needs > 3-foot gaps).
        * **Gripper:** 1-foot reach (must be very close to "pick up").
        * **Actions:** You can "move forward/backward [feet]",
            "strafe left/right [feet]", "ascend/descend [feet]",
            "turn left/right [degrees]", "approach [object]", "pick up [object]",
            "wait [seconds]", "unable to proceed".

        ### Your Task
        **High-Level Goal:** "{self.high_level_task}"

        **Recently Proposed Unsuccessful Action History:**
        {json.dumps(self.proposed_action_history) if self.proposed_action_history else "None yet."}

        {feedback_prompt_segment}

        ### Instructions
        1.  Analyze the provided image.
        2.  Consider your goal, history, and any critical feedback.
        3.  **CRITICAL RULE**: If the goal has a target object and that target object is clearly visible and appears to be within arm's reach (i.e. approximately within 1 foot or filling a large part of the view), you should propose the target action on that target object.
        4.  Otherwise, propose the *single best low-level action* to make progress (i.e. move, strafe, turn).
        5.  Be conservative. Small, safe steps are better.

        **Output:**
        Respond ONLY with the single action string. Do not add any other text.

        **Example Output:**
        move forward 5 feet
        """

        try:
            # API call
            action = self._llm.make_text_and_image_request(prompt, image)

            # Clean up response
            action = action.strip().replace("\"", "")
            self.proposed_action_history.append(action)
            return action

        except Exception as e:
            print(f"[Planner Error] An unexpected error occurred: {e}")
            return "wait 1 second" # A safe fallback action

    def approve_action(self, action):
        """
        Logs a successful action to the planner's history.
        """
        print(f"[Planner] Action Approved: {action}")
        self.proposed_action_history = []
        self.action_history.append(action)
        # TODO: Update self.progress_metric
        # e.g., self.progress_metric = self.calculate_progress(image, self.action_history)

    def check_task_completion(self) -> bool:
        """
        Determines if the high_level_task is effectively complete based
        on the semantic meaning of the action history.

        Returns:
            bool: True if complete, False otherwise.
        """
        # Might also be good to check robot internal state
        # i.e. GPS location, grasping open vs closed, other sensors, etc.

        if not self.action_history:
            return False

        prompt = f"""
        You are a Task Completion Judge for a robot.
        Determine if a robotic agent has successfully finished its High Level Task
        based *strictly* on the history of actions it has successfully executed.

        ### Input Data
        **High Level Task:** "{self.high_level_task}"
        **Action History (Chronological):** {json.dumps(self.action_history)}

        ### Judgment Rules
        1. Analyze the Goal: What is the final required state? (e.g., holding an object, arriving at a location).
        2. Analyze the History: Does the sequence of actions logically result in that state?
        3. Be Strict: If the last action was just "move forward" but the goal is "pick up cup", it is FALSE.
        4. Be Smart: If the goal is "Find the cup" and the last action is "approach cup", that might be TRUE. If the goal is "Retrieve cup" and last action is "pick up cup", that is TRUE.
        5. Assume successful completion of actions stated in action history and approve as long as they seem plausible and accomplish the high level task.

        ### Output
        Respond ONLY with a valid JSON object:
        {{
            "is_complete": <true | false>,
            "reasoning": "<brief explanation>"
        }}
        """

        try:
            # Since this is a text-only check, we use make_text_request
            # (assuming your factory supports a text-only method, otherwise pass a blank image)
            response_string = self._llm.make_text_request(prompt)

            result = extract_json(response_string)

            # Log the reasoning for debugging
            print(f"[Completion Check] {result['reasoning']}")

            return result.get("is_complete", False)

        except Exception as e:
            print(f"[Completion Check Error] {e}")
            return False


def robot_planning_loop(planner: LLM_Planner, checker: Feasibility_Checker, camera: Camera, high_level_task: str):
    """
    Runs main robot planning loop.
    """
    print(f"High-Level Task: {high_level_task}\n")
    high_level_task_complete = False
    planner.set_high_level_task(high_level_task) # This can be updated to get the next high level task for a larger loop
    checker.set_high_level_task(high_level_task)
    while not high_level_task_complete:
        print(f"--- IMAGE INDEX: {camera.frame_idx} ---")
        image = camera.get_next_frame()
        if image is None:
            raise Exception("High level task not complete")
        action_candidate = planner.propose_next_action(image)
        feasibility_feedback = checker.check_action_feasibility(image, action_candidate)
        print(f"Proposed action: {action_candidate}")

        feasible = feasibility_feedback.get("feasible", False)
        attempt = 0
        max_attempts = 3
        while not feasible and attempt < max_attempts:
            # print(f"Proposed action rejected.\nReason: {feasibility_feedback.get('reasoning', "None")}\nReplanning...")
            action_candidate = planner.propose_next_action(image, feasibility_feedback)
            feasibility_feedback = checker.check_action_feasibility(image, action_candidate)
            print(f"Proposed action: {action_candidate}")

            feasible = feasibility_feedback.get("feasible", False)
            attempt += 1
        if attempt >= max_attempts and not feasible:
            # Realistically should ask for help or terminate task
            # print(f"Proposed action rejected.\nReason: {feasibility_feedback.get('reasoning', "None")}")
            raise Exception("[Loop Error] Not able to find a next action within max_attempts")
        print("Proposed action approved")

        planner.approve_action(action_candidate)
        high_level_task_complete = planner.check_task_completion()

    print(f"High Level Task '{high_level_task}' complete")


if __name__ == "__main__":
    # Image 0: A hallway with a person blocking the path
    IMG_BLOCKED_HALLWAY = "https://www.shutterstock.com/shutterstock/videos/23979670/thumb/10.jpg?ip=x480"
    # Image 1: An open room
    IMG_OPEN_ROOM = "https://st.hzcdn.com/simgs/a85138450e8f9f97_14-4317/_.jpg"
    # Image 2: A desk with a cup on it, far away
    IMG_DESK_CUP_FAR = "https://cdn.shopify.com/s/files/1/0666/0220/5506/files/office-desk-with-computer-and-cup-of-coffee.jpg?v=1666070936"
    # Image 3: Close-up of the cup, in reach
    IMG_DESK_CUP_CLOSE = "https://images.unsplash.com/photo-1485808191679-5f86510681a2?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Y3VwJTIwb2YlMjBjb2ZmZWV8ZW58MHx8MHx8fDA%3D"

    # Initialize our main components
    planner = LLM_Planner(model="gpt-4o-mini")
    checker = Feasibility_Checker(model="gpt-4o-mini")
    camera = Camera(frames=[
        IMG_BLOCKED_HALLWAY,
        IMG_OPEN_ROOM,
        IMG_DESK_CUP_FAR,
        IMG_DESK_CUP_CLOSE])

    # Scenario
    high_level_task = "Go to the cup on the desk down the hallway and pick it up."

    print(f"\n--- STARTING SCENARIO ---")

    robot_planning_loop(planner, checker, camera, high_level_task)

"""
Results:
    Initial issue:
        Initializing LLM_Planner with model: gpt-4o-mini
        Initializing Feasibility_Checker with model: gpt-4o-mini

        --- STARTING SCENARIO ---
        High-Level Task: Go to the cup on the desk down the hallway and pick it up.

        Proposed action: move forward 3 feet
        Proposed action rejected.
        Reason: Infeasible: Path blocked by a person. I suggest you wait for the person to move.
        Replanning...
        Proposed action: wait 5 seconds
        Proposed action rejected.
        Reason: Infeasible: Path is blocked by a person. I suggest you wait until the person has moved.
        Replanning...
        Proposed action: wait 10 seconds
        Proposed action rejected.
        Reason: Infeasible: Path is blocked by a person. I suggest you wait until the person has moved.
        Replanning...
        Proposed action: wait 5 seconds
        Proposed action rejected.
        Reason: Infeasible: Path blocked by a person. I suggest you wait until the person has moved.
        Exception: [Loop Error] Not able to find a next action within max_attempts

    Added this to feasibility checker prompt: (EXCEPTION: If action is "wait", ignore obstacles. "Wait" is valid even if path is blocked).

Results 2:
    Initial Issue:
        Initializing LLM_Planner with model: gpt-4o-mini
        Initializing Feasibility_Checker with model: gpt-4o-mini

        --- STARTING SCENARIO ---
        High-Level Task: Go to the cup on the desk down the hallway and pick it up.

        --- IMAGE INDEX: 0 ---
        Proposed action: move forward 3 feet
        Proposed action rejected.
        Reason: Infeasible: Path blocked by a person. I suggest you wait until the person moves.
        Replanning...
        Proposed action: wait 5 seconds
        Proposed action approved
        [Planner] Action Approved: wait 5 seconds
        [Completion Check] The action history only includes 'wait 5 seconds', which does not contribute to reaching the goal of going to the cup and picking it up.
        --- IMAGE INDEX: 1 ---
        Proposed action: move forward 3 feet
        Proposed action approved
        [Planner] Action Approved: move forward 3 feet
        [Completion Check] The action history does not include any action related to reaching the cup or picking it up; it only shows waiting and moving forward, which does not fulfill the task requirements.
        --- IMAGE INDEX: 2 ---
        Proposed action: move forward 3 feet
        Proposed action rejected.
        Reason: Infeasible: Path blocked by the desk. I suggest you go above the desk.
        Replanning...
        Proposed action: ascend 2 feet
        Proposed action rejected.
        Reason: Infeasible: Path is blocked by the desk. I suggest you navigate around the desk.
        Replanning...
        Proposed action: strafe left 2 feet
        Proposed action rejected.
        Reason: Infeasible: Path blocked by the edge of the table. I suggest you move upwards or downwards to navigate around the table.
        Replanning...
        Proposed action: ascend 1 foot
        Proposed action rejected.
        Reason: Infeasible: Path is blocked by the desk. I suggest you navigate around the desk.
        [Loop Error] Not able to find a next action within max_attempts

    Updated feasibility checker prompt and make feasibility checker aware of high level goals
    Tested quite a few feasibility checker prompts

Results 3:
    Initial Issue:
        Initializing LLM_Planner with model: gpt-4o-mini
        Initializing Feasibility_Checker with model: gpt-4o-mini

        --- STARTING SCENARIO ---
        High-Level Task: Go to the cup on the desk down the hallway and pick it up.

        --- IMAGE INDEX: 0 ---
        Proposed action: move forward 3 feet
        Proposed action rejected.
        Reason: The path is blocked by a person walking down the hallway. Since there is a collision risk, the robot cannot move forward. A constructive suggestion would be to 'wait until the person has passed' or 'ascend to hover above the person to navigate around them.'
        Replanning...
        Proposed action: wait 5 seconds
        Proposed action approved
        [Planner] Action Approved: wait 5 seconds
        [Completion Check] The action history only includes 'wait 5 seconds', which does not contribute to reaching the goal of going to the cup and picking it up.
        --- IMAGE INDEX: 1 ---
        Proposed action: move forward 3 feet
        Proposed action rejected.
        Reason: The path is likely blocked by furniture (e.g., couch or table) in front of the robot. Since the cup is on a desk, the robot should try 'ascend' to hover over the cup or 'pick up cup' if it is within reach.
        Replanning...
        Proposed action: ascend 2 feet
        Proposed action approved
        [Planner] Action Approved: ascend 2 feet
        [Completion Check] The action history does not include any movement towards the desk or picking up the cup, which are necessary to complete the task.
        --- IMAGE INDEX: 2 ---
        Proposed action: move forward 3 feet
        Proposed action approved
        [Planner] Action Approved: move forward 3 feet
        [Completion Check] The robot has not reached the cup on the desk nor has it picked it up; it has only moved forward 3 feet and ascended 2 feet.
        --- IMAGE INDEX: 3 ---
        Proposed action: move forward 2 feet
        Proposed action approved
        [Planner] Action Approved: move forward 2 feet
        [Completion Check] The robot has not reached the cup on the desk nor has it picked it up; it has only moved forward and ascended.
        Exception: High level task not complete

    Modify LLM Planner away from bias toward navigation instead of manipulation
    Add some more exceptions into the feasibility checker
    Add another condition so that the high level task completion checker is less strict

Result 4:
    Initial Issue:
        Initializing LLM_Planner with model: gpt-4o-mini
        Initializing Feasibility_Checker with model: gpt-4o-mini

        --- STARTING SCENARIO ---
        High-Level Task: Go to the cup on the desk down the hallway and pick it up.

        --- IMAGE INDEX: 0 ---
        Proposed action: move forward 5 feet
        Proposed action rejected.
        Reason: The path is blocked by a person walking down the hallway. Since the goal is to pick up the cup on the desk, I suggest you 'wait' until the person has moved out of the way before proceeding.
        Replanning...
        Proposed action: wait 5 seconds
        Proposed action approved
        [Planner] Action Approved: wait 5 seconds
        [Completion Check] The action history only includes 'wait 5 seconds', which does not contribute to reaching the cup or picking it up. Therefore, the high level task is not completed.
        --- IMAGE INDEX: 1 ---
        Proposed action: move forward 5 feet
        Proposed action rejected.
        Reason: The proposed action to move forward 5 feet is INFEASIBLE because the path is likely blocked by a surface (desk or table) in front of the robot. Since the goal is to pick up a cup on that surface, I suggest trying to 'ascend' to hover over the cup or 'pick up cup' if it is already within reach.
        Replanning...
        Proposed action: ascend 2 feet
        Proposed action approved
        [Planner] Action Approved: ascend 2 feet
        [Completion Check] The action history does not include any movement towards the desk or the cup, nor does it include any action to pick up the cup. Therefore, the high level task of picking up the cup is not completed.
        --- IMAGE INDEX: 2 ---
        Proposed action: move forward 5 feet
        Proposed action rejected.
        Reason: The cup is on the desk, which is directly in front of the robot. However, if the cup is already within 1 foot, moving forward would pose a collision risk. Since the cup is likely within reach, I suggest you 'pick up the cup' instead of moving forward.
        Replanning...
        Proposed action: pick up the cup
        Proposed action approved
        [Planner] Action Approved: pick up the cup
        [Completion Check] The last action 'pick up the cup' successfully completes the high level task of picking up the cup on the desk.
        High Level Task 'Go to the cup on the desk down the hallway and pick it up.' complete

    There seems to be an issue that with the third image, the drone should move closer,
    however, it doesn't move close enough before attempting grap and the grasp is approved

Result 5 (same as 4 but with stronger model):
    Initializing LLM_Planner with model: gpt-4o-mini
    Initializing Feasibility_Checker with model: gpt-4o-mini

    --- STARTING SCENARIO ---
    High-Level Task: Go to the cup on the desk down the hallway and pick it up.

    --- IMAGE INDEX: 0 ---
    Proposed action: move forward 5 feet
    Proposed action rejected.
    Reason: Path is blocked by a person directly in front, which would result in a collision. Suggest waiting for the person to move or attempting to strafe if space allows.
    Replanning...
    Proposed action: wait 10 seconds
    Proposed action approved
    [Planner] Action Approved: wait 10 seconds
    [Completion Check] The action history only includes 'wait 10 seconds', which does not involve moving to the cup or picking it up. Therefore, the high level task is not completed.
    --- IMAGE INDEX: 1 ---
    Proposed action: turn right 90 degrees
    Proposed action approved
    [Planner] Action Approved: turn right 90 degrees
    [Completion Check] The action history only includes 'wait 10 seconds' and 'turn right 90 degrees', which do not result in reaching the cup on the desk or picking it up.
    --- IMAGE INDEX: 2 ---
    Proposed action: pick up cup
    Proposed action rejected.
    Reason: The cup is on the desk, which is a surface directly in front of the robot. The robot cannot move through the desk, but since the cup is on the desk, the 'Approach' rule allows moving forward to interact with the object. However, if the cup is already within reach, moving forward is infeasible due to collision risk. Suggestion: 'Object is already in reach. Do not move forward. I suggest you 'pick up cup'.'
    Replanning...
    Proposed action: approach cup
    Proposed action approved
    [Planner] Action Approved: approach cup
    [Completion Check] The action history does not include the action of picking up the cup, which is required to complete the high level task.
    --- IMAGE INDEX: 3 ---
    Proposed action: pick up cup
    Proposed action approved
    [Planner] Action Approved: pick up cup
    [Completion Check] The high level task is to go to the cup on the desk down the hallway and pick it up. The action history includes 'approach cup' and 'pick up cup', which indicates the robot has successfully reached the cup and picked it up, fulfilling the task requirements.
    High Level Task 'Go to the cup on the desk down the hallway and pick it up.' complete

The specifications here work successfully for the more powerful models
"""