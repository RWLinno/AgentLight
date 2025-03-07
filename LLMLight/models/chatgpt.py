import copy
from utils.my_utils import load_json, dump_json, get_state_detail, get_state_three_segment
import requests
import json
import time
import re
import csv
import io
import pandas as pd
import numpy as np

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "YOUR_KEY_HERE"
}

four_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3}
eight_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3, 'WTWL': 4, 'ETEL': 5, 'STSL': 6, 'NTNL': 7}
location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
location_dict_detail = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}
direction_dict = {"T": "through", "L": "left-turn", "R": "turn-right"}
direction_dict_ori = {"T": "through", "L": "turn-left", "R": "turn-right"}

phase_explanation_dict_detail = {"NTST": "- NTST: Northern and southern through lanes.",
                                 "NLSL": "- NLSL: Northern and southern left-turn lanes.",
                                 "NTNL": "- NTNL: Northern through and left-turn lanes.",
                                 "STSL": "- STSL: Southern through and left-turn lanes.",
                                 "ETWT": "- ETWT: Eastern and western through lanes.",
                                 "ELWL": "- ELWL: Eastern and western left-turn lanes.",
                                 "ETEL": "- ETEL: Eastern through and left-turn lanes.",
                                 "WTWL": "- WTWL: Western through and left-turn lanes."
                                }

incoming_lane_2_outgoing_road = {
    "NT": "South",
    "NL": "East",
    "ST": "North",
    "SL": "West",
    "ET": "West",
    "EL": "South",
    "WT": "East",
    "WL": "North"
}

class ChatGPTTLCS_Commonsense(object):
    def __init__(self, GPT_version, intersection, inter_name, phase_num, log_dir, dataset):
        # init road length
        roads = copy.deepcopy(intersection["roads"])
        self.inter_name = inter_name
        self.roads = roads
        self.length_dict = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
        for r in roads:
            self.length_dict[roads[r]["location"]] = int(roads[r]["length"])

        self.phases = four_phase_list if phase_num == 4 else eight_phase_list

        self.gpt_version = GPT_version
        self.last_action = "ETWT"
        self.system_prompt = load_json("./prompts/prompt_commonsense.json")["system_prompt"]
        self.state_action_prompt_file = f"{log_dir}/{dataset}-{self.inter_name}-{self.gpt_version}-{phase_num}_state_action_prompt_commonsense_no_calculation.json"
        self.error_file = f"{log_dir}/{dataset}-{self.inter_name}-{self.gpt_version}-{phase_num}_error_prompts_commonsense_no_calculation.json"
        self.state_action_prompt = []
        self.errors = []

        self.temp_action_logger = ""
        self.env = None  # Store the environment

    def choose_action(self, env):
        # Store the environment for later use
        self.env = env
        self.temp_action_logger = ""
        
        # Debug information about the environment
        print(f"\n===== CHATGPT AGENT ENVIRONMENT CHECK =====")
        print(f"Environment type: {type(env).__name__}")
        print(f"Environment has list_intersection: {hasattr(env, 'list_intersection')}")
        if hasattr(env, 'list_intersection'):
            print(f"Number of intersections: {len(env.list_intersection)}")
        print(f"Environment has intersection_dict: {hasattr(env, 'intersection_dict')}")
        print("============================================\n")
        
        # Get state details from the roads and environment
        try:
            state, state_incoming, avg_speed = get_state_detail(roads=self.roads, env=env)
            flow_num = 0
            for road in state:
                flow_num += state[road]["queue_len"] + sum(state[road]["cells"])

            if flow_num == 0:
                action_code = self.action2code("ETWT")
                self.state_action_prompt.append({"state": state, "prompt": [], "action": "ETWT"})
                dump_json(self.state_action_prompt, self.state_action_prompt_file)
                self.temp_action_logger = action_code
                return
        except Exception as e:
            print(f"Error getting state details: {e}")
            action_code = self.action2code("ETWT")
            self.state_action_prompt.append({"state": "Error", "prompt": [], "action": "ETWT"})
            dump_json(self.state_action_prompt, self.state_action_prompt_file)
            self.temp_action_logger = action_code
            return

        signal_text = ""

        # chain-of-thought
        retry_counter = 0
        while signal_text not in self.phases:
            if retry_counter > 10:
                signal_text = "ETWT"
                break
            try:
                state_txt = self.state2table(state)
                prompt = self.getPrompt(state_txt)
                data = {
                    "model": self.gpt_version,
                    "messages": prompt,
                    "max_tokens": 2048,
                    "temperature": 0.0
                }
                response = requests.post(url, headers=headers, data=json.dumps(data)).json()
                analysis = response['choices'][0]['message']['content']
                retry_counter += 1
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                matches = re.findall(signal_answer_pattern, analysis)
                signal_text = matches[-1] if matches else "ETWT"

            except Exception as e:
                print(f"Error in getting signal: {e}")
                self.errors.append({"error": str(e), "prompt": prompt})
                dump_json(self.errors, self.error_file)
                time.sleep(3)
                signal_text = "ETWT"  # Set default on error

        prompt.append({"role": "assistant", "content": analysis})
        action_code = self.action2code(signal_text)
        self.state_action_prompt.append({"state": state, "state_incoming": state_incoming, "prompt": prompt, "action": signal_text})
        dump_json(self.state_action_prompt, self.state_action_prompt_file)

        self.temp_action_logger = action_code
        self.last_action = signal_text

    '''
    ============ Class Utils ============
    '''

    def state2table(self, state):
        state_txt = ""
        for p in self.phases:
            lane_1 = p[:2]
            lane_2 = p[2:]
            queue_len_1 = int(state[lane_1]['queue_len'])
            queue_len_2 = int(state[lane_2]['queue_len'])

            seg_1_lane_1 = state[lane_1]['cells'][0]
            seg_2_lane_1 = state[lane_1]['cells'][1]
            seg_3_lane_1 = state[lane_1]['cells'][2] + state[lane_1]['cells'][3]

            seg_1_lane_2 = state[lane_2]['cells'][0]
            seg_2_lane_2 = state[lane_2]['cells'][1]
            seg_3_lane_2 = state[lane_2]['cells'][2] + state[lane_2]['cells'][3]

            state_txt += (f"Signal: {p}\n"
                          f"Allowed lanes: {phase_explanation_dict_detail[p][8:-1]}\n"
                          f"- Early queued: {queue_len_1} ({location_dict[lane_1[0]]}), {queue_len_2} ({location_dict[lane_2[0]]}), {queue_len_1 + queue_len_2} (Total)\n"
                          f"- Segment 1: {seg_1_lane_1} ({location_dict[lane_1[0]]}), {seg_1_lane_2} ({location_dict[lane_2[0]]}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                          f"- Segment 2: {seg_2_lane_1} ({location_dict[lane_1[0]]}), {seg_2_lane_2} ({location_dict[lane_2[0]]}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                          f"- Segment 3: {seg_3_lane_1} ({location_dict[lane_1[0]]}), {seg_3_lane_2} ({location_dict[lane_2[0]]}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n")

        return state_txt

    def getPrompt(self, state_txt):
        # fill information
        signals_text = ""
        for i, p in enumerate(self.phases):
            signals_text += phase_explanation_dict_detail[p] + "\n"

        prompt = [
            {"role": "system",
             "content": self.system_prompt},
            {"role": "user",
             "content": "A crossroad connects two roads: the north-south and east-west. The traffic light is located at "
                        "the intersection of the two roads. The north-south road is divided into two sections by the intersection: "
                        "the north and south. Similarly, the east-west road is divided into the east and west. Each section "
                        "has two lanes: a through and a left-turn lane. Each lane is further divided into three segments. "
                        "Segment 1 is the closest to the intersection. Segment 2 is in the middle. Segment 3 is the farthest. "
                        "In a lane, there may be early queued vehicles and approaching vehicles traveling in different segments. "
                        "Early queued vehicles have arrived at the intersection and await passage permission. Approaching "
                        "vehicles will arrive at the intersection in the future.\n\n"
                        "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two "
                        "specific lanes. The state of the intersection is listed below. It describes:\n"
                        "- The group of lanes relieving vehicles' flow under each traffic light phase.\n"
                        "- The number of early queued vehicles of the allowed lanes of each signal.\n"
                        "- The number of approaching vehicles in different segments of the allowed lanes of each signal.\n\n"
                        + state_txt +
                        "Please answer:\n"
                        "Which is the most effective traffic signal that will most significantly improve the traffic "
                        "condition during the next phase, which relieves vehicles' flow of the allowed lanes of the signal?\n\n"
                        "Note:\n"
                        "The traffic congestion is primarily dictated by the early queued vehicles, with the MOST significant "
                        "impact. You MUST pay the MOST attention to lanes with long queue lengths. It is NOT URGENT to "
                        "consider vehicles in distant segments since they are unlikely to reach the intersection soon.\n\n"
                        "Requirements:\n"
                        "- Let's think step by step.\n"
                        "- You can only choose one of the signals listed above.\n"
                        "- You must follow the following steps to provide your analysis: Step 1: Provide your analysis "
                        "for identifying the optimal traffic signal. Step 2: Answer your chosen signal.\n"
                        "- Your choice can only be given after finishing the analysis.\n"
                        "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>."
             },
        ]

        return prompt

    def action2code(self, action):
        code = self.phases[action]

        return code

