# HandyRL is licensed under The MIT License
# [See https://github.com/DeNA/HandyRL/blob/master/LICENSE for details]

# https://github.com/DeNA/HandyRL/blob/master/handyrl/envs/geister.py

import random
import itertools

import numpy as np


class Environment:
    X, Y = 'ABCDEF', '123456'
    BLACK, WHITE = 0, 1
    BLUE, RED = 0, 1
    C = 'BW'
    T = 'BR'
    P = {-1: '_', 0: 'B', 1: 'R', 2: 'b', 3: 'r', 4: '*'}
    # original positions to set pieces
    OPOS = [
        ['B2', 'C2', 'D2', 'E2', 'B1', 'C1', 'D1', 'E1'],
        ['E5', 'D5', 'C5', 'B5', 'E6', 'D6', 'C6', 'B6'],
    ]
    # goal positions
    GPOS = np.array([
        [(-1, 5), (6, 5)],
        [(-1, 0), (6, 0)]
    ], dtype=np.int32)

    D = np.array([(-1, 0), (0, -1), (0, 1), (1, 0)], dtype=np.int32)
    OSEQ = list(itertools.combinations([i for i in range(8)], 4))

    def __init__(self, args=None):
        super().__init__()
        self.args = args if args is not None else {}
        self.reset()

    def reset(self, args=None):
        self.game_args = args if args is not None else {}
        self.board = -np.ones((6, 6), dtype=np.int32)  # (x, y) -1 is empty
        self.color = self.BLACK
        self.turn_count = -2  # before setting original positions
        self.win_color = None
        self.piece_cnt = np.zeros(4, dtype=np.int32)
        self.board_index = -np.ones((6, 6), dtype=np.int32)
        self.piece_position = np.zeros((2 * 8, 2), dtype=np.int32)
        self.record = []
        self.captured_type = None
        self.layouts = {}

    def put_piece(self, piece, pos, piece_idx):
        self.board[pos[0], pos[1]] = piece
        self.piece_position[piece_idx] = pos
        self.board_index[pos[0], pos[1]] = piece_idx
        self.piece_cnt[piece] += 1

    def remove_piece(self, piece, pos):
        self.board[pos[0], pos[1]] = -1
        piece_idx = self.board_index[pos[0], pos[1]]
        self.board_index[pos[0], pos[1]] = -1
        self.piece_position[piece_idx] = np.array((-1, -1))
        self.piece_cnt[piece] -= 1

    def move_piece(self, piece, pos_from, pos_to):
        self.board[pos_from[0], pos_from[1]] = -1
        self.board[pos_to[0], pos_to[1]] = piece
        piece_idx = self.board_index[pos_from[0], pos_from[1]]
        self.board_index[pos_from[0], pos_from[1]] = -1
        self.board_index[pos_to[0], pos_to[1]] = piece_idx
        self.piece_position[piece_idx] = pos_to

    def set_pieces(self, c, seq_idx):
        # decide original positions
        chosen_seq = self.OSEQ[seq_idx]
        for idx in range(8):
            t = 0 if idx in chosen_seq else 1
            piece = self.colortype2piece(c, t)
            pos = self.str2position(self.OPOS[c][idx])
            self.put_piece(piece, pos, c * 8 + idx)

    def opponent(self, color):
        return self.BLACK + self.WHITE - color

    def onboard(self, pos):
        return 0 <= pos[0] and pos[0] < 6 and 0 <= pos[1] and pos[1] < 6

    def goal(self, c, pos):
        # check whether pos is goal position for c
        for g in self.GPOS[c]:
            if g[0] == pos[0] and g[1] == pos[1]:
                return True
        return False

    def colortype2piece(self, c, t):
        return c * 2 + t

    def piece2color(self, p):
        return -1 if p == -1 else p // 2

    def piece2type(self, p):
        return -1 if p == -1 else p % 2

    def rotate(self, pos):
        return np.array((5 - pos[0], 5 - pos[1]), dtype=np.int32)

    def position2str(self, pos):
        if self.onboard(pos):
            return self.X[pos[0]] + self.Y[pos[1]]
        else:
            return '**'

    def str2position(self, s):
        if s != '**':
            return np.array((self.X.find(s[0]), self.Y.find(s[1])), dtype=np.int32)
        else:
            return None

    def fromdirection2action(self, pos_from, d, c):
        if c == self.WHITE:
            pos_from = self.rotate(pos_from)
            d = 3 - d
        return d * 36 + pos_from[0] * 6 + pos_from[1]

    def action2from(self, a, c):
        pos1d = a % 36
        pos = np.array((pos1d / 6, pos1d % 6), dtype=np.int32)
        if c == self.WHITE:
            pos = self.rotate(pos)
        return pos

    def action2direction(self, a, c):
        d = a // 36
        if c == self.WHITE:
            d = 3 - d
        return d

    def action2to(self, a, c):
        return self.action2from(a, c) + self.D[self.action2direction(a, c)]

    def action2str(self, a, player):
        if a >= 4 * 6 * 6:
            return 's' + str(a - 4 * 6 * 6)

        c = player
        pos_from = self.action2from(a, c)
        pos_to = self.action2to(a, c)
        return self.position2str(pos_from) + self.position2str(pos_to)

    def str2action(self, s, player):
        if s[0] == 's':
            return 4 * 6 * 6 + int(s[1:])

        c = player
        pos_from = self.str2position(s[:2])
        pos_to = self.str2position(s[2:])

        if pos_to is None:
            # it should arrive at a goal
            for g in self.GPOS[c]:
                if ((pos_from - g) ** 2).sum() == 1:
                    diff = g - pos_from
                    for d, dd in enumerate(self.D):
                        if np.array_equal(dd, diff):
                            break
                    break
        else:
            # check action direction
            diff = pos_to - pos_from
            for d, dd in enumerate(self.D):
                if np.array_equal(dd, diff):
                    break

        return self.fromdirection2action(pos_from, d, c)

    def record_string(self):
        return ' '.join([self.action2str(a, i % 2) for i, a in enumerate(self.record)])

    def position_string(self):
        poss = [self.position2str(pos) for pos in self.piece_position]
        return ','.join(poss)

    def __str__(self):
        # output state
        def _piece(p):
            return p if p == -1 or self.layouts[self.piece2color(p)] >= 0 else 4

        s = '  ' + ' '.join(self.Y) + '\n'
        for i in range(6):
            s += self.X[i] + ' ' + ' '.join([self.P[_piece(self.board[i, j])] for j in range(6)]) + '\n'
        s += 'remained = B:%d R:%d b:%d r:%d' % tuple(self.piece_cnt) + '\n'
        s += 'turn = ' + str(self.turn_count) + ' color = ' + self.C[self.color]
        #s += 'record = ' + self.record_string()
        return s

    def _set(self, layout):
        self.layouts[self.color] = layout
        if layout < 0:
            layout = random.randrange(70)
        self.set_pieces(self.color, layout)
        self.color = self.opponent(self.color)
        self.turn_count += 1

    def play(self, action, _=None):
        # state transition
        if self.turn_count < 0:
            layout = action - 4 * 6 * 6
            return self._set(layout)

        ox, oy = self.action2from(action, self.color)
        nx, ny = self.action2to(action, self.color)
        piece = self.board[ox, oy]
        self.captured_type = None

        if not self.onboard((nx, ny)):
            # finish by goal
            self.remove_piece(piece, (ox, oy))
            self.win_color = self.color
        else:
            piece_cap = self.board[nx, ny]
            if piece_cap != -1:
                # capture opponent piece
                self.remove_piece(piece_cap, (nx, ny))
                if self.piece_cnt[piece_cap] == 0:
                    if self.piece2type(piece_cap) == self.BLUE:
                        # win by capturing all opponent blue pieces
                        self.win_color = self.color
                    else:
                        # lose by capturing all opponent red pieces
                        self.win_color = self.opponent(self.color)
                self.captured_type = self.piece2type(piece_cap)

            # move piece
            self.move_piece(piece, (ox, oy), (nx, ny))

        self.color = self.opponent(self.color)
        self.turn_count += 1
        self.record.append(action)

        if self.turn_count >= 200 and self.win_color is None:
            self.win_color = 2  # draw

    def diff_info(self, player):
        color = player
        played_color = (self.turn_count - 1) % 2
        info = {}
        if len(self.record) == 0:
            if self.turn_count > -2:
                info['set'] = self.layouts[played_color] if color == played_color else -1
        else:
            info['move'] = self.action2str(self.record[-1], played_color)
            if color == played_color and self.captured_type is not None:
                info['captured'] = self.T[self.captured_type]
        return info

    def update(self, info, reset):
        if reset:
            self.game_args = {**self.game_args, **info}
            self.reset(info)
        elif 'set' in info:
            self._set(info['set'])
        elif 'move' in info:
            action = self.str2action(info['move'], self.color)
            if 'captured' in info:
                # set color to captured piece
                pos_to = self.action2to(action, self.color)
                t = self.T.index(info['captured'])
                piece = self.colortype2piece(self.opponent(self.color), t)
                self.board[pos_to[0], pos_to[1]] = piece
            self.play(action)

    def turn(self):
        return self.players()[self.turn_count % 2]

    def observers(self):
        return self.players()

    def terminal(self):
        # check whether terminal state or not
        return self.win_color is not None

    def outcome(self):
        # return terminal outcomes
        outcomes = [0, 0]
        if self.win_color == self.BLACK:
            outcomes = [1, -1]
        elif self.win_color == self.WHITE:
            outcomes = [-1, 1]
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}

    def legal(self, action):
        if self.turn_count < 0:
            layout = action - 4 * 6 * 6
            return 0 <= layout < 70
        elif not 0 <= action < 4 * 6 * 6:
            return False

        pos_from = self.action2from(action, self.color)
        pos_to = self.action2to(action, self.color)

        piece = self.board[pos_from[0], pos_from[1]]
        c, t = self.piece2color(piece), self.piece2type(piece)
        if c != self.color:
            # no self piece on original position
            return False

        return self._legal(c, t, pos_from, pos_to)

    def _legal(self, c, t, pos_from, pos_to):
        if self.onboard(pos_to):
            # can move to cell if there isn't my piece
            piece_cap = self.board[pos_to[0], pos_to[1]]
            return self.piece2color(piece_cap) != c
        else:
            # can move to my goal
            return t == self.BLUE and self.goal(c, pos_to)

    def legal_actions(self, _=None):
        # return legal action list
        if self.turn_count < 0:
            return [4 * 6 * 6 + i for i in range(70)]
        actions = []
        for pos in self.piece_position[self.color*8:(self.color+1)*8]:
            if pos[0] == -1:
                continue
            t = self.piece2type(self.board[pos[0], pos[1]])
            for d in range(4):
                if self._legal(self.color, t, pos, pos + self.D[d]):
                    action = self.fromdirection2action(pos, d, self.color)
                    actions.append(action)

        return actions

    def players(self):
        return [0, 1]

    def observation(self, player=None):
        # state representation to be fed into neural networks
        turn_view = player is None or player == self.turn()
        color = self.color if turn_view else self.opponent(self.color)
        opponent = self.opponent(color)

        nbcolor = self.piece_cnt[self.colortype2piece(color,    self.BLUE)]
        nrcolor = self.piece_cnt[self.colortype2piece(color,    self.RED )]
        nbopp   = self.piece_cnt[self.colortype2piece(opponent, self.BLUE)]
        nropp   = self.piece_cnt[self.colortype2piece(opponent, self.RED )]

        s = np.array([
            1 if color == self.BLACK else 0,  # my color is black
            1 if turn_view           else 0,  # view point is turn player
            # the number of remained pieces
            *[(1 if nbcolor == i else 0) for i in range(1, 5)],
            *[(1 if nrcolor == i else 0) for i in range(1, 5)],
            *[(1 if nbopp   == i else 0) for i in range(1, 5)],
            *[(1 if nropp   == i else 0) for i in range(1, 5)]
        ]).astype(np.float32)

        blue_c = self.board == self.colortype2piece(color,    self.BLUE)
        red_c  = self.board == self.colortype2piece(color,    self.RED)
        blue_o = self.board == self.colortype2piece(opponent, self.BLUE)
        red_o  = self.board == self.colortype2piece(opponent, self.RED)

        b = np.stack([
            # board zone
            np.ones_like(self.board),
            # my/opponent's all pieces
            blue_c + red_c,
            blue_o + red_o,
            # my blue/red pieces
            blue_c,
            red_c,
            # opponent's blue/red pieces
            blue_o if player is None else np.zeros_like(self.board),
            red_o  if player is None else np.zeros_like(self.board)
        ]).astype(np.float32)

        if color == self.WHITE:
            b = np.rot90(b, k=2, axes=(1, 2))

        return {'scalar': s, 'board': b}



# symmetry

DIR_SYM = [3, 1, 2, 0]
SET_SYM = [
    # list(itertools.combinations([i for i in range(8)], 4))
    # list(itertools.combinations([3,2,1,0, 7,6,5,4], 4))
    0, 38, 37, 36, 35, 18, 17, 16, 15, 60,
    59, 57, 58, 56, 55, 8, 7, 6, 5, 50,
    49, 47, 48, 46, 45, 30, 29, 27, 28, 26,
    25, 68, 67, 66, 65, 4, 3, 2, 1, 44,
    43, 41, 42, 40, 39, 24, 23, 21, 22, 20,
    19, 64, 63, 62, 61, 14, 13, 11, 12, 10,
    9, 54, 53, 52, 51, 34, 33, 32, 31, 69
]

# inverse transformation check
assert False not in [DIR_SYM[d] == DIR_SYM.index(d) for d in range(4)]
assert False not in [SET_SYM[i] == SET_SYM.index(i) for i in range(70)]

def sym_obs(obs, sym_index):
    if obs is None:
        return None
    b = obs['board'] if sym_index == 0 else np.flip(obs['board'], 1)
    return {**obs, 'board': b}

def sym_policy(p, sym_index):
    if p is None:
        return None
    if sym_index == 1:
        p_move, p_set = p[:144], p[144:]
        p_move = np.flip(p_move.reshape(4, 6, 6), 1)  # x
        p_move = np.concatenate([p_move[d] for d in DIR_SYM], 0).reshape(144)  # dir
        p_set = p_set[SET_SYM]
        p = np.concatenate([p_move, p_set])
    return p

def sym_action(a, sym_index):
    if a is None:
        return None
    if sym_index == 1:
        if a < 144:
            d, x, y = a // 36, (a % 36) // 6, (a % 36) % 6
            a = DIR_SYM[d] * 36 + (5 - x) * 6 + y
        else:
            a = 144 + SET_SYM[a - 144]
    return a


class RandomAgent:
    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        return random.choice(actions)

    def observe(self, env, player, show=False):
        return [0.0]


def print_outputs(env, player, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(prob, v)
    else:
        if v is not None:
            print('v = %f' % v)
        if prob is not None:
            p = (prob * 1000).astype(int)
            if env.turn_count < 0:
                print('p =', p[144:])
            else:
                print('p =' + ' '.join([d.rjust(25) for d in ['up', 'left', 'right', 'down']])[4:])
                for x in range(6):
                    for d in range(4):
                         for y in range(6):
                              a = d * 36 + x * 6 + y if env.color == env.BLACK else (3 - d) * 36 + (5 - x) * 6 + (5 - y)
                              print('%4d' % p[a], end='')
                         if d < 3:
                             print(' |', end='')
                    print()


def apply_mask(x, ok_list):
    if x is None:
        return x
    mask = np.ones_like(x)
    mask[ok_list] = 0
    return x - mask * 1e32


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def map_r(x, callback_fn=None):
    # recursive map function
    if isinstance(x, (list, tuple, set)):
        return type(x)(map_r(xx, callback_fn) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, map_r(xx, callback_fn)) for key, xx in x.items())
    return callback_fn(x) if callback_fn is not None else None

def bimap_r(x, y, callback_fn=None):
    if isinstance(x, (list, tuple)):
        return type(x)(bimap_r(xx, y[i], callback_fn) for i, xx in enumerate(x))
    elif isinstance(x, dict):
        return type(x)((key, bimap_r(xx, y[key], callback_fn)) for key, xx in x.items())
    return callback_fn(x, y) if callback_fn is not None else None

def rotate(x, max_depth=1024):
    if max_depth == 0:
        return x
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], (list, tuple)):
            return type(x[0])(
                rotate(type(x)(xx[i] for xx in x), max_depth - 1)
                for i, _ in enumerate(x[0])
            )
        elif isinstance(x[0], dict):
            return type(x[0])(
                (key, rotate(type(x)(xx[key] for xx in x), max_depth - 1))
                for key in x[0]
            )
    elif isinstance(x, dict):
        x_front = x[list(x.keys())[0]]
        if isinstance(x_front, (list, tuple)):
            return type(x_front)(
                rotate(type(x)((key, xx[i]) for key, xx in x.items()), max_depth - 1)
                for i, _ in enumerate(x_front)
            )
        elif isinstance(x_front, dict):
            return type(x_front)(
                (key2, rotate(type(x)((key1, xx[key2]) for key1, xx in x.items()), max_depth - 1))
                for key2 in x_front
            )
    return x


class Agent:
    def __init__(self, model, temperature=0.0, temperature_decay=1.0, observation=True, symmetry=False):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.original_temperature = temperature
        self.temperature_decay = temperature_decay
        self.observation = observation
        self.symmetry = symmetry
        self.hidden = None
        self.temperature = None

    def reset(self, env, show=False):
        self.temperature = self.original_temperature
        self.hidden = map_r(self.model, lambda m: m.init_hidden([2] if self.symmetry else []))

    def plan(self, obs):
        if self.symmetry:
            obses = [sym_obs(obs, sym) for sym in range(2)]
            obs = bimap_r(obs, rotate(obses), lambda _, o: np.array(o))

        def forward(model, obs, hidden):
            outputs = model.inference(obs, hidden, batch_input=self.symmetry)
            hidden = outputs.pop('hidden', None)
            if self.symmetry:
                outputs_ = {}
                for k, o in outputs.items():
                    if k == 'policy':
                        outputs_[k] = np.mean([sym_policy(p, sym) for sym, p in enumerate(o)], 0)
                    else:
                        outputs_[k] = o.mean(0)
                outputs = outputs_
            return outputs, hidden

        if isinstance(self.model, list):  # ensemble
            outputs = {}
            for i, model in enumerate(self.model):
                outputs_, self.hidden[i] = forward(model, obs, self.hidden[i])
                for k, o in outputs_.items():
                    outputs[k] = outputs.get(k, []) + [o]
            for k, ol in outputs.items():
                outputs[k] = np.mean(ol, axis=0)
        else:
            outputs, self.hidden = forward(self.model, obs, self.hidden)
        return outputs

    def action(self, env, player, show=False):
        outputs = self.plan(env.observation(player))
        actions = env.legal_actions(player)
        p = apply_mask(outputs['policy'], actions)
        v = outputs.get('value', None)

        if show:
            print_outputs(env, player, softmax(p), v)

        if self.temperature < 1e-6:
            ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
            action = ap_list[0][0]
        else:
            action = random.choices(np.arange(len(p)), weights=softmax(p / self.temperature))[0]

        self.temperature *= self.temperature_decay
        return action

    def observe(self, env, player, show=False):
        v = None
        if self.observation:
            outputs = self.plan(env.observation(player))
            v = outputs.get('value', None)
            if show:
                print_outputs(env, player, None, v)
        return v if v is not None else [0.0]


class OnnxModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.ort_session = None
        self._open_session()

    def _open_session(self):
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'

        import onnxruntime
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        self.ort_session = onnxruntime.InferenceSession(self.model_path, sess_options=opts)

    def init_hidden(self, batch_size=None):
        hidden_inputs = [y for y in self.ort_session.get_inputs() if y.name.startswith('hidden')]
        if len(hidden_inputs) == 0:
            return None

        if batch_size is None:
            batch_size = []
        type_map = {
            'tensor(float)': np.float32,
            'tensor(int64)': np.int64,
        }
        hidden_tensors = [np.zeros(batch_size + list(y.shape[1:]), dtype=type_map[y.type]) for y in hidden_inputs]
        return hidden_tensors

    def inference(self, x, hidden=None, batch_input=False):
        # numpy array -> numpy array
        ort_inputs = {}
        ort_input_names = [y.name for y in self.ort_session.get_inputs()]

        def insert_input(y):
            y = y if batch_input else np.expand_dims(y, 0)
            ort_inputs[ort_input_names[len(ort_inputs)]] = y
        map_r(x, lambda y: insert_input(y))
        if hidden is not None:
            map_r(hidden, lambda y: insert_input(y))
        ort_outputs = self.ort_session.run(None, ort_inputs)
        if not batch_input:
            ort_outputs = [o.squeeze(0) for o in ort_outputs]

        ort_output_names = [y.name for y in self.ort_session.get_outputs()]
        outputs = {name: ort_outputs[i] for i, name in enumerate(ort_output_names)}

        hidden_outputs = []
        for k in list(outputs.keys()):
            if k.startswith('hidden'):
                hidden_outputs.append(outputs.pop(k))
        if len(hidden_outputs) == 0:
            hidden_outputs = None

        outputs = {**outputs, 'hidden': hidden_outputs}
        return outputs



class GatAgent:
    def __init__(self, agent):
        self.agent = agent

    @staticmethod
    def gat2hrl_set(env, s_action):
        ps = s_action[4:]
        seq = tuple('ABCDEFGH'.index(pchar) for pchar in ps)
        return 144 + env.OSEQ.index(seq)

    @staticmethod
    def gat2hrl_action(env, s_action, player):
        pchar, dchar = s_action[4], s_action[6]
        index = 'ABCDEFGH'.index(pchar)
        d = 'WSNE'.index(dchar)
        if player == 1:
            d = 3 - d
        pos = env.piece_position[player * 8 + index]
        return env.fromdirection2action(pos, d, player)

    @staticmethod
    def gat2hrl_action_result_info(env, action, s_ret, player):
        info = {}
        if 0 <= action < 144:
            info['move'] = env.action2str(action, player)
            if s_ret[2] in 'BR' and '*' not in info['move']:
                info['captured'] = s_ret[2]
        else:
            info['set'] = action - 144
        return info

    @staticmethod
    def gat2hrl_board_info(env, s_board, player):
        # opponent piece position was updated
        s_board = s_board[4:]  # remove 'MOV?'
        info = {}
        for idx in range(8, 16):
            p = s_board[3 * idx: 3 * (idx + 1)]
            pos = int(p[0]), 5 - int(p[1])
            if pos[0] >= 6:
                pos = -1, -1
            if player == 1:
                idx = idx - 8
                if env.onboard(pos):
                    pos = env.rotate(pos)

            if not (pos == env.piece_position[idx]).all():
                pos_from = env.piece_position[idx]
                pos_to = pos
                s = env.position2str(pos_from) + env.position2str(pos_to)
                info['move'] = s
                break
        return info

    @staticmethod
    def hrl2gat_set(env, action):
        seq = env.OSEQ[action - 144]
        red_seq = [i for i in range(8) if i not in seq]
        a_gat = 'SET:' + ''.join(['ABCDEFGH'[index] for index in red_seq])
        return a_gat

    @staticmethod
    def hrl2gat_action(env, action):
        pos_from = env.action2from(action, env.color)
        d = env.action2direction(action, env.BLACK)

        # change goal direction
        pos_to = env.action2to(action, env.color)
        if env.goal(env.color, pos_to):
            d = 2  # NORTH

        piece_index = env.board_index[pos_from[0], pos_from[1]]
        pchar = 'ABCDEFGH'[piece_index % 8]
        a_gat = 'MOV:' + pchar + ',' + 'WSNE'[d]
        return a_gat

    @staticmethod
    def hrl2gat_board(env, player):
        s_board = ''
        P = 'ABCDEFGHabcdefgh'
        for idx in range(16):
            p = [player, 1 - player][idx // 8]
            piece_idx = p * 8 + idx % 8
            piece_type = 0 if (idx % 8) in env.OSEQ[env.layouts[p]] else 1
            pos = env.piece_position[piece_idx]
            if env.goal(p, pos):
                s_pos = '88'
                cchar = 'br'[piece_type]
            elif not env.onboard(pos):
                s_pos = '99'
                cchar = 'br'[piece_type]
            else:
                if player == 1:
                    pos = env.rotate(pos)
                s_pos = str(pos[0]) + str(5 - pos[1])
                if p == player:
                    cchar = 'BR'[piece_type]
                else:
                    cchar = 'u'
            s_board += s_pos + cchar

        return s_board


# Game Loop

def send(sock, s):
    print('<< %s' % s)
    sock.send((s + '\r\n').encode('utf-8'))

recv_buf = []
def recv(sock):
    global recv_buf
    if len(recv_buf) == 0:
        recv_buf += sock.recv(256).decode('utf-8').split('\r\n')[:-1]
    s, recv_buf = recv_buf[0], recv_buf[1:]
    print('>> %s' % s)
    return s


def gat_loop(agent, port, host):
    env = Environment()
    player_mine = 0 if port == 10000 else 1
    result = [0, 0, 0]

    while True:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        _ = recv(sock)  # 'SET?'
        env.update({}, reset=True)
        agent.reset(env, show=True)

        if player_mine == 0:
            action = agent.action(env, player_mine, show=True)
            env.update({'set': action - 144}, reset=False)
            agent.observe(env, player_mine, show=True)
            env.update({'set': -1}, reset=False)
        else:
            agent.observe(env, player_mine, show=True)
            env.update({'set': -1}, reset=False)
            action = agent.action(env, player_mine, show=True)
            env.update({'set': action - 144}, reset=False)
            agent.observe(env, player_mine, show=True)  # only for WHITE
        s_action = GatAgent.hrl2gat_set(env, action)
        send(sock, s_action)
        _ = recv(sock)

        while True:
            s_board = recv(sock)
            if not s_board.startswith('MOV?'):
                break  # gameend
            info = GatAgent.gat2hrl_board_info(env, s_board, player_mine)
            if player_mine == 1 or env.turn_count > 0:
                env.update(info, reset=False)

            print('*** MY TURN *** ')
            print(env)
            action = agent.action(env, player_mine, show=True)
            print('selected %s' % env.action2str(action, player_mine))
            s_action = GatAgent.hrl2gat_action(env, action)
            send(sock, s_action)
            s_ret = recv(sock)
            info = GatAgent.gat2hrl_action_result_info(env, action, s_ret, player_mine)
            env.update(info, reset=False)

            if not env.terminal():
                print('*** OPPONENT TURN *** ')
                print(env)
                agent.observe(env, player_mine, show=True)

        result['WDL'.index(s_board[0])] += 1
        print('result =', result)

        sock.close()
        import time
        time.sleep(0.2)



def debug_games():
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            actions = e.legal_actions()
            print([e.action2str(a, e.turn()) for a in actions])
            e.play(random.choice(actions))
        print(e)
        print(e.outcome())


if __name__ == '__main__':
    #debug_games()

    import sys
    port = int(sys.argv[1])
    host = sys.argv[2] if len(sys.argv) >= 3 else 'localhost'

    #agent = RandomAgent()
    models = [OnnxModel('dummy.onnx')]
    agent = Agent(models, temperature=1.0, observation=False, symmetry=True)
    gat_loop(agent, port, host)
