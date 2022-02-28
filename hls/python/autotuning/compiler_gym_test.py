import gym

env = gym.make("llvm-autophase-ic-v0")
print(env.action_space.dtype)
print(env.action_space.n)
benchmark = env.make_benchmark(
    '/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.opt.vitis.bkup.ll')
env.reset(benchmark=benchmark)

best_opt = (0, "")
episode_reward = 0
for i in range(1, 101):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break
    episode_reward += reward
    print(f"Step {i}, quality={episode_reward:.3%}")
    if episode_reward > best_opt[0]:
        print(env.commandline())
        env.write_ir("/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.opt.vitis.ll")
        best_opt = (episode_reward, env.commandline())
