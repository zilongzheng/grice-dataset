import random


dialoge_types = [
    'explicit', 'close-but', ''
]

def select_subtopic(oracle):
    return random.choice(['agent_act', 'obj_loc', 'agent_loc', 'obj_scale'])


def select_utterance_type(oracle, )

class STAOG:
    def __init__(self, oracle) -> None:
        self.t = 0
        self.oracle = oracle

    
    def generate_next_turn(self, d_type=None):

        st = select_subtopic(self.oracle)

        ut = select_utterance_type(self.oracle, st)