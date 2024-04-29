import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from common.buffers import OfflineBuffer
from utils.train_tools import soft_target_update, evaluate
from utils import log_tools


class plkc_Agent:
    def __init__(self,
                 env,
                 data_buffer: OfflineBuffer,
                 critic_net1: torch.nn.Module,
                 critic_net2: torch.nn.Module,
                 perturbation_net: torch.nn.Module,
                 # cvae_net: torch.nn.Module,  # generation model

                 #************** cgan ****************
                 generator_net: torch.nn.Module,
                 discriminator_net: torch.nn.Module,
                 cgan_lr=3e-4,
                 loss_fn=torch.nn.BCELoss(),
                 #************************************

                 critic_lr=1e-3,
                 per_lr=1e-3,
                 # cvae_lr=1e-3,

                 gamma=0.99,
                 tau=0.005,
                 lmbda=0.75,  # used for double clipped double q-learning
                 batch_size=64,
                 latent_dim=4,
                 max_train_step=200000,
                 log_interval=1000,
                 eval_freq=5000,
                 train_id="sac_Pendulum_test",
                 resume=False,  # if True, train from last checkpoint
                 device='cpu',
                 ):

        self.data_buffer = data_buffer
        self.device = torch.device(device)
        self.env = env
        self.critic_net1 = critic_net1.to(self.device)
        self.critic_net2 = critic_net2.to(self.device)
        self.target_critic_net1 = copy.deepcopy(self.critic_net1).to(self.device)
        self.target_critic_net2 = copy.deepcopy(self.critic_net2).to(self.device)
        self.perturbation_net = perturbation_net.to(self.device)
        self.target_perturbation_net = copy.deepcopy(self.perturbation_net).to(self.device)
        # self.cvae_net = cvae_net.to(self.device)

        # ************** cgan ****************
        self.generator_net = generator_net.to(self.device)
        self.discriminator_net = discriminator_net.to(self.device)
        self.g_optim = torch.optim.Adam(self.generator_net.parameters(), lr=1e-2, betas=(0.4, 0.8), weight_decay=0.0001)
        self.d_optim = torch.optim.Adam(self.discriminator_net.parameters(), lr=cgan_lr, betas=(0.4, 0.8), weight_decay=0.0001)
        self.loss_fn = loss_fn.to(self.device)
        self.labels_one = torch.ones(batch_size, 1).to(self.device)
        self.labels_zero = torch.zeros(batch_size, 1).to(self.device)
        # ************************************

        self.critic_optimizer1 = torch.optim.Adam(self.critic_net1.parameters(), lr=critic_lr)
        self.critic_optimizer2 = torch.optim.Adam(self.critic_net2.parameters(), lr=critic_lr)
        self.perturbation_optimizer = torch.optim.Adam(self.perturbation_net.parameters(), lr=per_lr)
        # self.cvae_optimizer = torch.optim.Adam(self.cvae_net.parameters(), lr=cvae_lr)

        self.gamma = gamma
        self.tau = tau
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.max_train_step = max_train_step
        self.eval_freq = eval_freq
        self.train_step = 0

        self.resume = resume  # whether load checkpoint start train from last time

        # log dir and interval
        self.log_interval = log_interval
        self.result_dir = os.path.join(log_tools.ROOT_DIR, "run/results", train_id)
        log_tools.make_dir(self.result_dir)
        self.checkpoint_path = os.path.join(self.result_dir, "checkpoint.pth")
        self.tensorboard_writer = log_tools.TensorboardLogger(self.result_dir)

    def choose_action(self, obs, eval=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).reshape(1, -1).repeat(100, 1).to(self.device)
            z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
            generated_action = self.generator_net(z, obs).to(self.device)
            perturbed_action = self.perturbation_net(obs, generated_action)
            q1 = self.critic_net1(obs, perturbed_action)
            ind = q1.argmax(dim=0)
        return perturbed_action[ind].cpu().data.numpy().flatten()

    def train(self):

        # Sample
        batch = self.data_buffer.sample()
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        """
        CVAE Loss (the generation model)
        """
        # recon_action, mu, log_std = self.cvae_net(obs, acts)
        # cvae_loss = self.cvae_net.loss_function(recon_action, acts, mu, log_std)
        #
        # self.cvae_optimizer.zero_grad()
        # cvae_loss.backward()
        # self.cvae_optimizer.step()

        """
        CGAN Loss (the generation model)
        """
        # z:高斯噪声
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)

        # 生成器G
        pred_acts = self.generator_net(z, obs)   # 生成器G：输入(噪声+label)   输出：pred_acts
        self.g_optim.zero_grad()
        recons_loss = torch.abs(pred_acts - acts).mean()
        g_loss = recons_loss*0.05 + self.loss_fn(self.discriminator_net(pred_acts, obs), self.labels_one)
        g_loss.backward()
        self.g_optim.step()

        # 辨别器D
        self.d_optim.zero_grad()
        real_loss = self.loss_fn(self.discriminator_net(acts, obs), self.labels_one)    # 真实图片 与 真实标签的BCELOSS
        fake_loss = self.loss_fn(self.discriminator_net(pred_acts.detach(), obs), self.labels_zero)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optim.step()

        """
        Critic Loss
        """
        with torch.no_grad():
            # generated_action = self.cvae_net.decode(next_obs, z_device=self.device)
            z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
            generated_action = self.generator_net(z, next_obs).to(self.device)

            for i in range(9):
                z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
                temp_act = self.generator_net(z, next_obs).to(self.device)
                generated_action = torch.cat([generated_action, temp_act]).to(self.device)

            # print(generated_action.shape) # torch.Size([1000, 6])
            # generate 10 actions for every next_obs
            next_obs = torch.repeat_interleave(next_obs, repeats=10, dim=0).to(self.device)

            # perturb the generated action
            perturbed_action = self.target_perturbation_net(next_obs, generated_action)
            # compute target Q value of perturbed action
            target_q1 = self.target_critic_net1(next_obs, perturbed_action)
            target_q2 = self.target_critic_net2(next_obs, perturbed_action)
            # soft clipped double q-learning
            target_q = self.lmbda * torch.min(target_q1, target_q2) + (1. - self.lmbda) * torch.max(target_q1, target_q2)
            # take max over each action sampled from the generation and perturbation model
            target_q = target_q.reshape(obs.shape[0], 10, 1).max(1)[0].squeeze(1)
            target_q = rews + self.gamma * (1. - done) * target_q

        # compute current Q
        current_q1 = self.critic_net1(obs, acts).squeeze(1)
        current_q2 = self.critic_net2(obs, acts).squeeze(1)
        # compute critic loss
        critic_loss1 = F.mse_loss(current_q1, target_q)
        critic_loss2 = F.mse_loss(current_q2, target_q)

        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        """
        Perturbation Loss
        """

        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        generated_action_ = self.generator_net(z, obs).to(self.device)
        perturbed_action_ = self.perturbation_net(obs, generated_action_)
        perturbation_loss = -self.critic_net1(obs, perturbed_action_).mean()

        self.perturbation_optimizer.zero_grad()
        perturbation_loss.backward()
        self.perturbation_optimizer.step()

        """
        Update target networks4     
        """
        soft_target_update(self.critic_net1, self.target_critic_net1, tau=self.tau)
        soft_target_update(self.critic_net2, self.target_critic_net2, tau=self.tau)
        soft_target_update(self.perturbation_net, self.target_perturbation_net, tau=self.tau)

        self.train_step += 1
        return d_loss.cpu().item(),g_loss.cpu().item(), (critic_loss1 + critic_loss2).cpu().item(), perturbation_loss.cpu().item()

    def learn(self):
        if self.resume:
            self.load_agent_checkpoint()
        else:
            # delete tensorboard log file
            log_tools.del_all_files_in_dir(self.result_dir)

        while self.train_step < (int(self.max_train_step)):
            d_loss, g_loss, critic_loss, perturbation_loss = self.train()

            if self.train_step % self.eval_freq == 0:
                avg_reward, avg_length = evaluate(agent=self, episode_num=10, show=False)
                self.tensorboard_writer.log_eval_data({"eval_episode_length": avg_length,
                                                       "eval_episode_reward": avg_reward}, self.train_step)

            if self.train_step % self.log_interval == 0:
                self.store_agent_checkpoint()
                self.tensorboard_writer.log_train_data({"d_loss": d_loss,
                                                        "g_loss": g_loss,
                                                        "critic_loss": critic_loss,
                                                        "perturbation_loss": perturbation_loss
                                                        }, self.train_step)

    def store_agent_checkpoint(self):
        checkpoint = {
            "critic_net1": self.critic_net1.state_dict(),
            "critic_net2": self.critic_net2.state_dict(),
            "perturbation_net": self.perturbation_net.state_dict(),
            # "cvae_net": self.cvae_net.state_dict(),
            "generator_net:": self.generator_net.state_dict(),
            "discriminator_net:": self.discriminator_net.state_dict(),

            "critic_optimizer1": self.critic_optimizer1.state_dict(),
            "critic_optimizer2": self.critic_optimizer2.state_dict(),
            "perturbation_optimizer": self.perturbation_optimizer.state_dict(),
            # "cvae_optimizer": self.cvae_optimizer.state_dict(),
            "g_optim": self.g_optim.state_dict(),
            "d_optim": self.d_optim.state_dict(),

            "train_step": self.train_step,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_agent_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)  # can load gpu's data on cpu machine
        self.critic_net1.load_state_dict(checkpoint["critic_net1"])
        self.critic_net2.load_state_dict(checkpoint["critic_net2"])
        self.perturbation_net.load_state_dict(checkpoint["perturbation_net"])
        # self.cvae_net.load_state_dict(checkpoint["cvae_net"])
        self.generator_net.load_state_dict(checkpoint['generator_net'])
        self.discriminator_net.load_state_dict(checkpoint["discriminator_net"])

        self.critic_optimizer1.load_state_dict(checkpoint["critic_optimizer1"])
        self.critic_optimizer2.load_state_dict(checkpoint["critic_optimizer2"])
        self.perturbation_optimizer.load_state_dict(checkpoint["perturbation_optimizer"])
        # self.cvae_optimizer.load_state_dict(checkpoint["cvae_optimizer"])
        self.g_optim.load_state_dict(checkpoint["g_optim"])
        self.d_optim.load_state_dict(checkpoint["d_optim"])

        self.train_step = checkpoint["train_step"]

        print("load checkpoint from \"" + self.checkpoint_path +
              "\" at " + str(self.train_step) + " time step")



