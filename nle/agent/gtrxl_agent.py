# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is an example self-contained agent running NLE based on MonoBeast.

import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
from collections import deque
import math

# Necessary for multithreading.
os.environ["OMP_NUM_THREADS"] = "1"
H_DIM = 32

try:
    import torch
    from torch import multiprocessing as mp
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    logging.exception(
        "PyTorch not found. Please install the agent dependencies with "
        '`pip install "nle[agent]"`'
    )

import gym  # noqa: E402

import nle  # noqa: F401, E402
from nle import nethack  # noqa: E402
from nle.agent import vtrace  # noqa: E402

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--env", type=str, default="NetHackScore-v0",
                    help="Gym environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--model_path", default="~/torchbeast/",
                    help="Path to model checkpoint")
parser.add_argument("--num_actors", default=16, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=1000000000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--tf_layers", default=4, type=int,
                    help='number of transformer layers')
parser.add_argument("--tf_nheads", default=4, type=int,
                    help='number of attention heads')
parser.add_argument("--dim_feedforward", default=32, type=int,
                    help="transformer feedforward dim")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0001,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="tanh",
                    choices=["abs_one", "none", "tanh"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.0002,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.000001, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)


def get_tf_params(flags):
    params = {
        'nhead': flags.tf_nheads,
        'dim_feedforward': flags.dim_feedforward,
        'num_encoder_layers': flags.tf_layers,
        'unroll_len': flags.unroll_length
    }
    return params


def nested_map(f, n):
    if isinstance(n, tuple) or isinstance(n, list):
        return n.__class__(nested_map(f, sn) for sn in n)
    elif isinstance(n, dict):
        return {k: nested_map(f, v) for k, v in n.items()}
    else:
        return f(n)


def compute_baseline_loss(advantages, pad_mask=None):
    if pad_mask is not None:
        advantages = advantages * pad_mask
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits, pad_mask=None):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    l = policy * log_policy
    if pad_mask is not None:
        l *= pad_mask.unsqueeze(-1)
    return torch.sum(l)


def compute_policy_gradient_loss(logits, actions, advantages, pad_mask=None):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    if pad_mask is not None:
        cross_entropy = cross_entropy * pad_mask
    return torch.sum(cross_entropy * advantages.detach())


def create_env(name, *args, **kwargs):
    return gym.make(name, observation_keys=("glyphs", "blstats"), *args, **kwargs)


def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers,
    agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        gym_env = create_env(flags.env, savedir=flags.rundir)
        env = ResettingEnvironment(gym_env)
        env_output = env.initial()
        initial_agent_state = model.initial_state(batch_size=1)
        agent_output, *_ = model(env_output, initial_agent_state[:flags.tf_layers+2])
        list_ff_data = None

        while True:
            index = free_queue.get()
            if index is None:
                break

            if agent_state_buffers[index][-1].item():
                list_ff_data = None
                for i, t in enumerate(initial_agent_state):
                    agent_state_buffers[index][i][...] = t

            if list_ff_data is not None:
                for i in range(len(list_ff_data)):
                    k, v, low_k_attn = list_ff_data[i]
                    k = k[:, -flags.unroll_length:, ...]
                    v = v[:, -flags.unroll_length:, ...]
                    low_k_attn = low_k_attn[:, :, -flags.unroll_length:]
                    list_ff_data[i] = [k, v, low_k_attn]

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i in range(flags.tf_layers):
                agent_state_buffers[index][i][...] = agent_state_buffers[index][i+flags.tf_layers+2][...]
            agent_state_buffers[index][flags.tf_layers][...] = agent_state_buffers[index][flags.tf_layers+1]
            agent_state_buffers[index][flags.tf_layers+1].fill_(1)

            # Do new rollout.
            for t in range(flags.unroll_length):
                agent_state_buffers[index][flags.tf_layers + 1][t, ...] = 0
                with torch.no_grad():
                    agent_output, agent_delta_state, list_ff_data = model(env_output,
                                                                          agent_state_buffers[index][:flags.tf_layers+2],
                                                                          list_ff_data, t)

                for i, delta in enumerate(agent_delta_state):
                    time_delta = delta.shape[0]
                    agent_state_buffers[index][i+flags.tf_layers+2][t:t+time_delta, ...] = delta
                env_output = env.step(agent_output["action"])
                agent_state_buffers[index][-1][...] = env_output['done'].item()

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                if agent_state_buffers[index][-1].item():
                    break

            full_queue.put(index)

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers,
    agent_state_buffers,
    lock=threading.Lock(),
):
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    agent_state = [torch.cat([agent_state_buffers[m][k] for m in indices], dim=1) for k in range(flags.tf_layers+2)]
    agent_state[-1] = torch.cat([agent_state[-1], torch.ones_like(agent_state[-1][:1])])

    for m in indices:
        free_queue.put(m)
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in agent_state
    )
    return batch, agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        pad_mask = initial_agent_state[-1]  # (T, B)
        learner_outputs, *_ = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}
        pad_mask = 1 - pad_mask[:-1]

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "tanh":
            clipped_rewards = torch.tanh(rewards/100)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
            pad_mask=pad_mask
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"],
            pad_mask=pad_mask
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"],
            pad_mask=pad_mask
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def collate_fn(initial_agent_st, initial_agent_done):
    maxT = max(done.shape[0] for done in initial_agent_done)
    pad_done = torch.zeros((1, 1), dtype=torch.uint8)
    pad_st = torch.zeros((1, 1, H_DIM))

    for i, (st, done) in enumerate(zip(initial_agent_st, initial_agent_done)):
        initial_agent_done[i] = torch.cat([pad_done.tile(maxT-done.shape[0], 1), done], dim=0)
        initial_agent_st[i] = torch.cat([pad_st.tile(maxT-st.shape[0], 1, 1), st], dim=0)

    return torch.cat(initial_agent_st, dim=1), torch.cat(initial_agent_done, dim=1)


def create_buffers(flags, observation_space, num_actions, num_overlapping_steps=1):
    size = (flags.unroll_length + num_overlapping_steps,)

    # Get specimens to infer shapes and dtypes.
    samples = {k: torch.from_numpy(v) for k, v in observation_space.sample().items()}

    specs = {
        key: dict(size=size + sample.shape, dtype=sample.dtype)
        for key, sample in samples.items()
    }
    specs.update(
        reward=dict(size=size, dtype=torch.float32),
        done=dict(size=size, dtype=torch.bool),
        episode_return=dict(size=size, dtype=torch.float32),
        episode_step=dict(size=size, dtype=torch.int32),
        policy_logits=dict(size=size + (num_actions,), dtype=torch.float32),
        baseline=dict(size=size, dtype=torch.float32),
        last_action=dict(size=size, dtype=torch.int64),
        action=dict(size=size, dtype=torch.int64),
    )
    buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.zeros(**specs[key]).share_memory_())
    return buffers


def create_state_buffers(flags):
    initial_agent_state_buffers = list()
    for _ in range(flags.num_buffers):
        initial_agent_state_buffers.append([
        *[torch.zeros((flags.unroll_length, 1, H_DIM)).share_memory_() for _ in range(flags.tf_layers)],
        torch.ones((flags.unroll_length, 1), dtype=torch.uint8).share_memory_(),
        torch.ones((flags.unroll_length, 1), dtype=torch.uint8).share_memory_(),
        *[torch.zeros((flags.unroll_length, 1, H_DIM)).share_memory_() for _ in range(flags.tf_layers)],
        torch.zeros((1,), dtype=torch.uint8).share_memory_()
        ])
    return initial_agent_state_buffers


def _format_observations(observation, keys=("glyphs", "blstats")):
    observations = {}
    for key in keys:
        entry = observation[key]
        entry = torch.from_numpy(entry)
        entry = entry.view((1, 1) + entry.shape)  # (...) -> (T,B,...).
        observations[key] = entry
    return observations


class StateBuffer:
    def __init__(self, maxlen):
        self.curr_beg_ind = torch.zeros(1, dtype=torch.int32)
        self.maxlen = maxlen
        self.st_buffer = deque([torch.zeros((1, 1, H_DIM), dtype=torch.uint8) for _ in range(maxlen)], maxlen=maxlen)
        self.done_buffer = deque([torch.ones((1, 1)) for _ in range(maxlen)], maxlen=maxlen)

    def get_all(self):
        return torch.cat([*self.st_buffer]), torch.cat([*self.done_buffer]), self.curr_beg_ind

    def append(self, agent_state):
        st_tensor, done_tensor = agent_state
        if self.done_buffer[0]:
            self.curr_beg_ind[:] = 1
        else:
            self.curr_beg_ind += 1
        self.st_buffer.append(st_tensor)
        self.done_buffer.append(done_tensor)


class ResettingEnvironment:
    """Turns a Gym environment into something that can be step()ed indefinitely."""

    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)

        result = _format_observations(self.gym_env.reset())
        result.update(
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )
        return result

    def step(self, action):
        observation, reward, done, unused_info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            observation = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        result = _format_observations(observation)

        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        result.update(
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )
        return result

    def close(self):
        self.gym_env.close()


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    flags.savedir = os.path.expandvars(os.path.expanduser(flags.savedir))

    rundir = os.path.join(
        flags.savedir, "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    )

    if not os.path.exists(rundir):
        os.makedirs(rundir)
    logging.info("Logging results to %s", rundir)

    symlink = os.path.join(flags.savedir, "latest")
    try:
        if os.path.islink(symlink):
            os.remove(symlink)
        if not os.path.exists(symlink):
            os.symlink(rundir, symlink)
        logging.info("Symlinked log directory: %s", symlink)
    except OSError:
        raise

    logfile = open(os.path.join(rundir, "logs.tsv"), "a", buffering=1)
    checkpointpath = os.path.join(rundir, "model.tar")

    flags.rundir = rundir

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags.env)
    observation_space = env.observation_space
    action_space = env.action_space
    del env  # End this before forking.

    tf_params = get_tf_params(flags)
    model = Net(observation_space, action_space.n, flags.use_lstm, tf_params=tf_params)
    buffers = create_buffers(flags, observation_space, model.num_actions)

    model.share_memory()

    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    # Add initial RNN state.
    initial_agent_state_buffers = create_state_buffers(flags)

    actor_processes = []
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
            name="Actor-%i" % i,
        )
        actor.start()
        actor_processes.append(actor)

    learner_model = Net(observation_space, action_space.n, flags.use_lstm, tf_params=tf_params).to(
        device=flags.device
    )
    learner_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logfile.write("# Step\t%s\n" % "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        while step < flags.total_steps:
            batch, agent_state = get_batch(
                flags, free_queue, full_queue, buffers, initial_agent_state_buffers
            )
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler
            )
            with lock:
                logfile.write("%i\t" % step)
                logfile.write("\t".join(str(stats[k]) for k in stat_keys))
                logfile.write("\n")
                step += T * B

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn,
            name="batch-and-learn-%d" % i,
            args=(i,),
            daemon=True,  # To support KeyboardInterrupt below.
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        logging.warning("Quitting.")
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    logfile.close()


def test(flags, num_episodes=10):
    print("Work in progress")


class RandomNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm):
        super(RandomNet, self).__init__()
        del observation_shape, use_lstm
        self.num_actions = num_actions
        self.theta = torch.nn.Parameter(torch.zeros(self.num_actions))

    def forward(self, inputs, core_state):
        # print(inputs)
        T, B, *_ = inputs["observation"].shape
        zeros = self.theta * 0
        # set logits to 0
        policy_logits = zeros[None, :].expand(T * B, -1)
        # set baseline to 0
        baseline = policy_logits.sum(dim=1).view(-1, B)

        # sample random action
        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1).view(
            T, B
        )
        policy_logits = policy_logits.view(T, B, self.num_actions)
        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )

    def initial_state(self, batch_size):
        return ()


def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)


class Crop(nn.Module):
    """Helper class for NetHackNet below."""

    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[
            None, :
        ].expand(self.height_target, -1)
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[
            :, None
        ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )


class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.w_z = nn.Linear(d_model, d_model, bias=True)
        self.u_z = nn.Linear(d_model, d_model, bias=False)
        self.w_r = nn.Linear(d_model, d_model, bias=True)
        self.u_r = nn.Linear(d_model, d_model, bias=False)
        self.w_h = nn.Linear(d_model, d_model, bias=True)
        self.u_h = nn.Linear(d_model, d_model, bias=False)
        self.init_bias()

    def init_bias(self):
        self.w_z.bias.data.fill_(-2)
        self.w_r.bias.data.fill_(0)
        self.w_h.bias.data.fill_(0)

    def forward(self, skip, proc):
        z = torch.sigmoid(self.w_z(proc) + self.u_z(skip))
        r = torch.sigmoid(self.w_r(proc) + self.u_r(skip))
        h = torch.tanh(self.w_h(proc) + self.u_h(r * skip))
        g = (1 - z) * skip + z * h
        return g


class RMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, batch_first=False,
                 device=None, dtype=None, layer_norm_eps=1e-5, pos_mtx=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RMHA, self).__init__()
        self.embed_dim = embed_dim
        self.bias = bias

        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.pos_proj_weight = nn.Parameter(torch.empty((self.head_dim, embed_dim), **factory_kwargs))
        self.V_vec = nn.Parameter(torch.empty((self.head_dim, 1), **factory_kwargs))
        self.U_vec = nn.Parameter(torch.empty((self.head_dim, 1), **factory_kwargs))

        if bias:
            self.bias_q = nn.Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.bias_k = nn.Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.bias_pos = nn.Parameter(torch.empty(self.head_dim, **factory_kwargs))
        else:
            self.register_parameter('bias_q', None)
            self.register_parameter('bias_k', None)
            self.register_parameter('bias_v', None)
            self.register_parameter('bias_pos', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self._reset_parameters()

        pos_l = pos_mtx.shape[0]
        pos_encoding_i = pos_mtx.unsqueeze(0)  # (L, d) -> (1, L, d)
        pos_encoding_i = pos_encoding_i.repeat(pos_l, 1, 1)  # (L, L, d)
        pos_encoding_j = pos_mtx.unsqueeze(1)  # (L, d) -> (L, 1, d)
        pos_encoding_j = pos_encoding_j.repeat(1, pos_l, 1)  # (L, L, d)
        relative_pos_enc = (pos_encoding_i - pos_encoding_j)  # (L, L, d)
        self.register_buffer("relative_pos_enc", relative_pos_enc)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.xavier_uniform_(self.pos_proj_weight)
        nn.init.xavier_uniform_(self.V_vec)
        nn.init.xavier_uniform_(self.U_vec)

        if self.bias:
            nn.init.constant_(self.bias_q, 0.)
            nn.init.constant_(self.bias_k, 0.)
            nn.init.constant_(self.bias_v, 0.)
            nn.init.constant_(self.bias_pos, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, inp, prev_window, key_padding_mask=None, attn_mask=None):
        prev_window = prev_window.detach()
        if self.batch_first:
            inp.transpose_(0, 1)
            prev_window.transpose_(0, 1)
        e_inp = torch.cat([prev_window, inp], dim=0)

        q = F.linear(inp, self.q_proj_weight, self.bias_q)  # (Nt, B, d)
        k = F.linear(e_inp, self.k_proj_weight, self.bias_k)  # (Ns, B, d)
        v = F.linear(e_inp, self.v_proj_weight, self.bias_v)  # (Ns, B, d)

        q_len, kv_len, bsz = q.shape[0], k.shape[0], v.shape[1]
        q = q.view(q_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (B`, Nt, d`)
        k = k.view(kv_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (B`, Ns, d`)
        v = v.view(kv_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (B`, Ns, d`)

        relative_pos_enc = self.relative_pos_enc[kv_len-q_len:kv_len, :kv_len, ...]  # (Ns, Nt, d)
        r = F.linear(relative_pos_enc, self.pos_proj_weight, self.bias_pos).unsqueeze(0)  # (1, Ns, Nt, d')

        qk_attn = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B', Nt, Ns)
        k_attn = (self.U_vec.view(1, 1, -1) * k).sum(dim=-1).unsqueeze(1).repeat(1, q_len, 1)  # (B`, Nt, Ns)
        qr_attn = (q.unsqueeze(2) * r).sum(dim=-1)  # (B', Nt, Ns)
        r_attn = (self.V_vec.view(1, 1, 1, -1) * r).sum(dim=-1)  # (B', Nt, Ns)
        attn = qk_attn + k_attn + qr_attn + r_attn  # (B', Nt, Ns)

        mask = attn_mask.unsqueeze(0) + key_padding_mask  # (B, Nt, Ns)
        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(bsz * self.num_heads, q_len, kv_len)  # (B', Nt, Ns)

        attn += mask  # (B', Nt, Ns)

        soft_attn = F.softmax(attn, dim=-1)  # (B', Nt, Ns)
        out = torch.bmm(soft_attn, v).view(bsz, q_len, self.embed_dim).transpose(0, 1)  # (Nt, B, d)

        out = self.out_norm(inp + self.out_proj(out))
        ff_data = [k, v, k_attn[:, -1:]]
        return out, ff_data

    def fast_forward(self, inp, prev_ff_data, key_padding_mask=None):
        prev_k, prev_v, low_k_attn = prev_ff_data
        assert not self.batch_first
        assert inp.shape[0] == 1
        bsz = inp.shape[1]
        prev_kv_len = prev_k.shape[1]

        new_q = F.linear(inp, self.q_proj_weight, self.bias_q).view(1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (B', 1, d)
        new_k = F.linear(inp, self.k_proj_weight, self.bias_k).view(1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (B', 1, d)
        new_v = F.linear(inp, self.v_proj_weight, self.bias_v).view(1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (B', 1, d)

        k = torch.cat([prev_k, new_k], dim=1)
        v = torch.cat([prev_v, new_v], dim=1)  # (B', prev_kv_len+1, d')

        # compute delta r
        low_r = self.relative_pos_enc[prev_kv_len+1, :prev_kv_len+1]  # (prev_kv_len+1, d)
        low_r = F.linear(low_r, self.pos_proj_weight, self.bias_pos).view(1, 1, prev_kv_len+1, -1)  # (1, 1, prev_kv_len+1, d')

        # compute new qk_attn
        low_qk_attn = torch.bmm(new_q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B', 1, prev_kv_len+1)

        # compute new k_attn
        new_low_k_attn = (self.U_vec.view(1, 1, -1) * new_k).sum(dim=-1).unsqueeze(1)  # (B', 1, 1)
        low_k_attn = torch.cat([low_k_attn, new_low_k_attn], dim=-1)  # (B', 1, prev_kv_len+1)

        # compute new qr_attn
        low_qr_attn = (new_q.unsqueeze(2) * low_r).sum(dim=-1)  # (B', 1, prev_kv_len+1)

        # compute new r_attn
        low_r_attn = (self.V_vec.view(1, 1, 1, -1) * low_r).sum(dim=-1)  # (1, 1, prev_kv_len+1)

        # compute new attn
        low_attn = low_qk_attn + low_k_attn + low_qr_attn + low_r_attn

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, -1:, :]
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(bsz * self.num_heads, 1, prev_kv_len+1)
            low_attn += key_padding_mask  # (B', 1, prev_kv_len+1)

        soft_attn = F.softmax(low_attn, dim=-1)  # (B', 1, prev_kv_len+1)
        out = torch.bmm(soft_attn, v)  # (B', 1, d')
        out = out.view(bsz, 1, self.embed_dim).transpose(0, 1)  # (1, B, d)
        out = self.out_norm(inp + self.out_proj(out))

        ff_data = [k, v, low_k_attn]
        return out, ff_data


class BaseEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, pos_mtx, dim_feedforward=2048, dropout=0., activation="relu",
                 layer_norm_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BaseEncoderLayer, self).__init__()
        self.self_attn = RMHA(embed_dim=d_model, num_heads=nhead, bias=True,
                              layer_norm_eps=layer_norm_eps, pos_mtx=pos_mtx,
                              **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = torch.relu

        self.gate1 = Gate(d_model)
        self.gate2 = Gate(d_model)

    def forward(self, src, prev_window, src_mask=None, src_key_padding_mask=None, ff_data=None):
        norm_src = self.norm1(src)
        if ff_data is not None:
            attn_out, ff_data = self.self_attn.fast_forward(norm_src, ff_data,
                                                            src_key_padding_mask)
        else:
            attn_out, ff_data = self.self_attn(norm_src, prev_window, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        src = self.gate1(src, self.dropout1(self.activation(attn_out)))

        norm_src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(norm_src))))
        src = self.gate2(src, self.dropout2(self.activation(src2)))
        return src, ff_data


class BaseEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, unroll_len):
        super(BaseEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.unroll_len = unroll_len
        pe = self.generate_pos_enc_mtx(unroll_len*3)
        mask = self.generate_square_subsequent_mask(unroll_len*3)
        self.register_buffer('attn_mask', mask)

        self.encoding_layers = nn.ModuleList([
            BaseEncoderLayer(d_model, nhead=num_heads, pos_mtx=pe, dim_feedforward=dim_feedforward)
            for _ in range(num_layers)
            ])

    def forward(self, core_input, done, prev_window_states, list_ff_data=None, t=None):
        prev_len = prev_window_states[0].shape[0]
        src_mask = self.attn_mask[-t:, -t-prev_len:]
        src_key_padding_mask = self.generate_pad_mask(done, t, prev_len)

        out = core_input
        new_window_states = list()
        if list_ff_data is None:
            list_ff_data = [None for _ in range(self.num_layers)]

        for i in range(self.num_layers):
            new_window_states.append(out)
            out, ff_data = self.encoding_layers[i](out, prev_window_states[i], src_mask=src_mask,
                                                   src_key_padding_mask=src_key_padding_mask,
                                                   ff_data=list_ff_data[i])
            list_ff_data[i] = ff_data

        return out, new_window_states, list_ff_data


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_pad_mask(self, done, core_len, prev_len):
        done_len = done.shape[0]
        assert done_len == core_len + prev_len, "error in pad mask gen"
        mask = list()
        for v in done.transpose(0, 1):  # (L,)
            m = v.unsqueeze(0).float().repeat(core_len, 1)
            m = m.masked_fill(m > 0, float('-inf'))
            mask.append(m)
        mask = torch.stack(mask, dim=0)
        return mask

    def generate_pos_enc_mtx(self, unroll_len):
        pe = list()
        for i in range(0, self.d_model, 2):
            s = torch.sin(torch.arange(unroll_len, dtype=torch.float32) / (10000 ** ((2 * i) / self.d_model)))
            c = torch.cos(torch.arange(unroll_len, dtype=torch.float32) / (10000 ** ((2 * (i + 1)) / self.d_model)))
            pe.extend((s, c))
        pe = torch.stack(pe, dim=1)
        return pe


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=32, nhead=5, dim_feedforward=32, num_encoder_layers=4, unroll_len=80):
        super(TransformerEncoder, self).__init__()
        self.net = BaseEncoder(num_layers=num_encoder_layers, d_model=d_model,
                               num_heads=nhead, dim_feedforward=dim_feedforward,
                               unroll_len=unroll_len)

    def forward(self, inp, done_vector, prev_window_states, list_ff_data, t):
        out, new_window_states, list_ff_data = self.net(core_input=inp, done=done_vector,
                                                        prev_window_states=prev_window_states,
                                                        list_ff_data=list_ff_data, t=t)
        return out, new_window_states, list_ff_data


class NetHackNet(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_actions,
        use_lstm,
        embedding_dim=32,
        crop_dim=9,
        num_layers=5,
        tf_params=None
    ):
        super(NetHackNet, self).__init__()

        self.glyph_shape = observation_shape["glyphs"].shape
        self.blstats_size = observation_shape["blstats"].shape[0]

        self.num_actions = num_actions
        self.use_lstm = use_lstm

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = H_DIM

        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # CNN crop model.
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        out_dim = self.k_dim
        # CNN over full glyph map
        out_dim += self.H * self.W * Y

        # CNN crop model.
        out_dim += self.crop_dim ** 2 * Y

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        self.unroll_length = flags.unroll_length
        self.tf_layers = 1
        if self.use_lstm:
            params = {
                'd_model': H_DIM,
                'nhead': 4,
                'dim_feedforward': 32,
                'num_encoder_layers': 4,
                'unroll_len': 80,
            }
            if tf_params:
                params.update(tf_params)
            self.tf_layers = params['num_encoder_layers']
            self.core = TransformerEncoder(**params)

        self.policy = nn.Linear(self.h_dim, self.num_actions)
        self.baseline = nn.Linear(self.h_dim, 1)

    def initial_state(self, batch_size=1):
        init_state = (
            *(torch.zeros((self.unroll_length, batch_size, H_DIM)) for _ in range(self.tf_layers)),
            torch.ones((self.unroll_length, batch_size)),
            torch.ones((self.unroll_length, batch_size)),
            *(torch.zeros((self.unroll_length, batch_size, H_DIM)) for _ in range(self.tf_layers)),
            torch.zeros((1,), dtype=torch.uint8)
        )
        init_state[self.tf_layers+1][0, ...] = 0
        return init_state

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs, past_window, list_ff_data=None, t=None):
        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]

        # -- [T x B x F]
        blstats = env_outputs["blstats"]

        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # -- [B' x F]
        blstats = blstats.view(T * B, -1).float()

        # -- [B x H x W]
        glyphs = glyphs.long()
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]
        # TODO ???
        # coordinates[:, 0].add_(-1)

        # -- [B x F]
        blstats = blstats.view(T * B, -1).float()
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)

        assert blstats_emb.shape[0] == T * B

        reps = [blstats_emb]

        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)

        # print("crop", crop)
        # print("at_xy", glyphs[:, coordinates[:, 1].long(), coordinates[:, 0].long()])

        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)

        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb)

        # -- [B x K']
        crop_rep = crop_rep.view(T * B, -1)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # glyphs_emb = self.embed(glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)

        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st = self.fc(st)

        if self.use_lstm:
            prev_window_states, prev_done, curr_done = past_window[:-2], past_window[-2], past_window[-1]
            core_input = st.view(T, B, -1)
            t = t + 1 if t else core_input.shape[0]
            done = torch.cat([prev_done, curr_done[:t]], dim=0)
            core_output, new_window_states, list_ff_data = self.core(core_input, done,
                                                                     prev_window_states,
                                                                     list_ff_data, t)
            core_output = core_output[-T:].view(T*B, -1)
            new_window_states = [state[-T:] for state in new_window_states]
        else:
            core_output = st
            new_window_states = list()

        # -- [B x A]
        policy_logits = self.policy(core_output)
        # -- [B x A]
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            new_window_states, list_ff_data
        )


Net = NetHackNet


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
