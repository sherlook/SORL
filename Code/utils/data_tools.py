import numpy as np
from mlagents.trainers import demo_loader

try:
    import d4rl
except ImportError:
    print('No module named "d4rl" , and you can install in https://github.com/rail-berkeley/d4rl')

try:
    import d4rl_atari
except ImportError:
    print('No module named "d4rl_atari" , and you can install in https://github.com/takuseno/d4rl-atari')


def get_d4rl_dataset(env, get_num=None) -> dict:
    """
    d4rl dataset: https://github.com/rail-berkeley/d4rl
    install: pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
    :param get_num: how many data get form dataset
    """
    dataset = d4rl.qlearning_dataset(env)
    if get_num is None:
        data = dict(
            obs=dataset['observations'],
            acts=dataset['actions'],
            rews=dataset['rewards'],
            next_obs=dataset['next_observations'],
            done=dataset['terminals']
        )
    else:
        data_num = dataset['actions'].shape[0]
        ind = np.random.choice(data_num, size=get_num, replace=False)
        data = dict(
            obs=dataset['observations'][ind],
            acts=dataset['actions'][ind],
            rews=dataset['rewards'][ind],
            next_obs=dataset['next_observations'][ind],
            done=dataset['terminals'][ind]
        )

    return data


def get_d4rl_dataset_atari(env) -> dict:
    """
    d4rl atari dataset: https://github.com/takuseno/d4rl-atari
    install: pip install git+https://github.com/takuseno/d4rl-atari
    """
    dataset = env.get_dataset()
    data = dict(
        obs=dataset['observations'],
        acts=dataset['actions'],
        rews=dataset['rewards'],
        done=dataset['terminals']
    )

    return data


def LoadDataFromUnityDemo(demo_path):
    _, info_action_pairs, _ = demo_loader.load_demonstration(demo_path)

    # print("len(info_action_pairs): ", len(info_action_pairs))
    # print("info_action_pairs[0]: ", info_action_pairs[0])
    
    # print(info_action_pairs[0].agent_info.observations[0].float_data.data)
    # print(info_action_pairs[0].action_info.continuous_actions)

    observations = []
    actions = []
    rewards = []
    dones = []
    next_observations = []


    
    obs_pre = np.array(info_action_pairs[0].agent_info.observations[0].float_data.data, dtype=np.float32)
    
    # 将两个obs拼接在一起（射线+位置）
    # obs_pre = np.hstack((np.array(info_action_pairs[0].agent_info.observations[0].float_data.data, dtype=np.float32),
    #                     np.array(info_action_pairs[0].agent_info.observations[1].float_data.data, dtype=np.float32)))


    print("obs_pre:", obs_pre.size)
    done_pre = info_action_pairs[0].agent_info.done

    print("len(info_action_pairs): ",len(info_action_pairs))
    for info_action_pair in info_action_pairs[1:]:
        agent_info = info_action_pair.agent_info
        action_info = info_action_pair.action_info

        obs = np.array(agent_info.observations[0].float_data.data, dtype=np.float32)
        
        # 将两个obs拼接在一起（射线+位置）
        # obs = np.hstack((np.array(info_action_pairs[0].agent_info.observations[0].float_data.data, dtype=np.float32),
        #                 np.array(info_action_pairs[0].agent_info.observations[1].float_data.data, dtype=np.float32)))

        # print("data.obs.size:", obs.size)
        rew = agent_info.reward
        act = np.array(action_info.continuous_actions, dtype=np.float32)
        done = agent_info.done
        if not done_pre:
            observations.append(obs_pre)
            actions.append(act)
            rewards.append(rew)
            dones.append(done)
            next_observations.append(obs)

        obs_pre = obs
        done_pre = done

    data = dict(
                obs=np.array(observations, dtype=np.float32),
                acts=np.array(actions, dtype=np.float32),
                rews=np.array(rewards, dtype=np.float32),
                next_obs=np.array(next_observations, dtype=np.float32),
                done=np.array(dones, dtype=np.float32),
            )

    return data
