from arch import *
from config import *
from dataset import *
from torch.utils.data import Dataset, DataLoader
import random
from torchvision.utils import save_image
import os
from collections import deque
import torch.optim as optim
import copy
from tqdm import tqdm
import json
import torch.nn.functional as F
from utils import save_img_mnist

class Agent_Attack():
    def __init__(self, load=False, device="cpu", batch_size_att=256, batch_size=4096):
        torch.cuda.current_device()
        self.batch_size_att = batch_size_att
        self.batch_size = batch_size
        self.EPS = EPS
        self.epsilon = EPSILON
        self.device = device
        classifier = CONV_MNIST().eval()
        classifier.load_state_dict(torch.load(r"trained_model/mnist_cnn_best.pth"))
        self.classifier = classifier.cuda()

        self.img_size = IMG_SIZE
        self.max_iter = MAX_ITER

        self.grid_size = 2
        self.grid_map = self.img_size // self.grid_size

        self.input_size = self.img_size * self.img_size * 2
        self.output_size = self.grid_map * self.grid_map
        
        self.policy_net =  DQN_CNN2(self.input_size, self.output_size).cuda().train()
        
        self.memory = deque(maxlen=150000)
        self.success_memory = deque(maxlen=5000)
        self.criterion = nn.MSELoss().cuda()
        self.TARGET_UPDATE =TARGET_UPDATE
        self.GAMMA = GAMMA

        self.num = 0
        
        if load:
            print("Loading Trained model")
            self.policy_net.load_state_dict(torch.load(r"/kaggle/input/model-trien-vong-2/model_0_trrenvong_2.pth"))

        self.target_net = DQN_CNN2(self.input_size, self.output_size).cuda()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=5e-6)
        train_dataset = get_dataset('mnist', split='train')
        self.train_dataset = train_dataset
        test_dataset = get_dataset('mnist', split='test')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size_att, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        self.action_lists = []
        self.reward_lists = []
        self.loss_lists = []

    # def reward(self, img, prob, orig_img, label):
    #     new_prob = self.classifier_img(img)
    #     img_diff = torch.norm(img - orig_img) 
    #     prob_diff = torch.norm(prob - new_prob) 
    #     new_label = new_prob.argmax()
    #     if new_label != label[0]:
    #         return torch.tensor(2).float()
    #     # return prob_diff - 2*img_diff
    #     return torch.tensor(0).float()
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            actions = torch.randint(0, self.output_size, (self.batch_size_att, 1))
            return actions
        with torch.no_grad():
            prob = self.policy_net(state.cuda()).cpu()
            actions = prob.argmax(1)
            # print(actions)
            return actions
        
    def optimize_model(self):
        success_batch_size = min(len(self.success_memory), int(0.05 * self.batch_size))
        normal_batch_size = self.batch_size - success_batch_size
        # normal_batch_size = BATCH_SIZE
        if len(self.memory) < normal_batch_size:
            return
        if len(self.success_memory) == 0:
            return
        success_transitions = random.sample(self.success_memory, success_batch_size)
        transitions = random.sample(self.memory, normal_batch_size)
        success_batch = Transition(*zip(*success_transitions))
        batch = Transition(*zip(*transitions))
        # state_batch = torch.stack(batch.state).cuda().detach()
        # action_batch = torch.cat(batch.action).unsqueeze(1).detach()
        # reward_batch = torch.cat(batch.reward).detach()
        # next_state_batch = torch.stack(batch.next_state).cuda().detach()
        # done_batch = torch.tensor(batch.done).detach()
        state_batch = torch.concatenate([torch.stack(batch.state).cuda().detach(), torch.stack(success_batch.state).cuda().detach()])
        action_batch = torch.concatenate([torch.cat(batch.action).unsqueeze(1).detach(), torch.cat(success_batch.action).unsqueeze(1).detach()])
        reward_batch = torch.concatenate([torch.cat(batch.reward).detach(), torch.cat(success_batch.reward).detach()])
        next_state_batch = torch.concatenate([torch.stack(batch.next_state).cuda().detach(), torch.stack(success_batch.next_state).cuda().detach()])
        done_batch = torch.concatenate([torch.tensor(batch.done).detach(), torch.tensor(success_batch.done).detach()])
        
        state_action_values = self.policy_net(state_batch.squeeze(1)).cpu()
        state_action_values = state_action_values.gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch.squeeze(1)).cpu()
        next_state_values = next_state_values.max(1)[0].detach()
        expected_state_action_values = torch.where(done_batch, reward_batch, (next_state_values * self.GAMMA) + reward_batch)

        # print(expected_state_action_values)

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    
    def make_action(self, orgi_img, mask, grid, e = 0.2):

        new_img = orgi_img.clone()
        mask[grid] = -mask[grid]

        mask_img = self.create_mask(mask)
        new_img = new_img + mask_img * e

        new_img = torch.clamp(new_img, 0, 1)

        return new_img, mask
    
    def classifier_img(self, img):
        img = img.cuda()
        prob = self.classifier(img).cpu()
        img = img.cpu()
        return prob
    
    def create_mask(self, grid):
        # Đưa grid vào tensor và xác định kích thước
        grid_tensor = torch.tensor(grid).view(self.grid_map, self.grid_map)
        
        # Phóng to mỗi ô của grid_tensor thành vùng tương ứng trong mask
        mask = grid_tensor.repeat_interleave(self.grid_size, dim=0).repeat_interleave(self.grid_size, dim=1)
        
        # Định dạng mask thành (1, img_size, img_size)
        mask = mask.unsqueeze(0).to(dtype=torch.float32)
        
        return mask
    
    def create_list_mask(self, grids):
        list_mask = torch.zeros(self.batch_size_att, 1, self.img_size, self.img_size)
        for i in range(len(grids)):
            list_mask[i] = self.create_mask(grids[i])
        return list_mask

    def train(self):
        k = 0
        batch_norm = nn.BatchNorm2d(1)
        total_success = 0
        his = []
        for img, label in tqdm(self.train_loader):
            if len(label) != self.batch_size_att:
                continue
            k += 1
            print(len(self.memory))
            print(self.epsilon)
            orig_prob = self.classifier_img(img)
            masks = torch.randint(0, 2, (self.batch_size_att, self.grid_map * self.grid_map)).float() * 2 - 1
            dones = [False] * self.batch_size_att
            for i in range(self.batch_size_att):
                if label[i] != orig_prob[i].argmax():
                    dones[i] = True
                continue
                save_image(img[i], f"original/{k}_{i}.png")
            # orig_img = img.clone()
            # list_mask = self.create_list_mask(masks)
            state = torch.cat([img.view(self.batch_size_att, -1), masks], dim=1)
            new_img = img.clone()
            for j in range(self.max_iter):
                actions = self.select_action(state)
                self.action_lists.append(actions)
                for i in range(self.batch_size_att):
                    if dones[i]:
                        continue
                    new_img[i], masks[i] = self.make_action(img[i], masks[i], actions[i], 0.1)
                prob = self.classifier_img(new_img)
                # list_mask = self.create_list_mask(masks)
                next_state = torch.cat([img.view(self.batch_size_att, -1), masks], dim=1)
                for i in range(self.batch_size_att):
                    if dones[i]:
                        continue
                    if label[i] != prob[i].argmax():
                        dones[i] = True
                        self.success_memory.append(Transition(state[i], torch.tensor([actions[i]]), next_state[i], torch.tensor([2]), True))
                        with open(r"logs_success.txt", "a") as f:
                            f.write(f"{k}_{i}: {label[i]} -> {prob[i].argmax()} with {torch.norm(new_img[i] -img[i])}, prob_label: {orig_prob[i][label[i]].item()} -> {prob[i][label[i]].item()}, prob_max: {prob[i].max().item()}\n")
                    else:
                        # if random.random() < 0.1:
                        self.memory.append(Transition(state[i], torch.tensor([actions[i]]), next_state[i], torch.tensor([0]), False))
                        # self.memory.append(Transition(state[i], torch.tensor([actions[i]]), next_state[i], torch.tensor([0]), False))
                if all(dones):
                    break
                
                if j == self.max_iter - 1:
                    for i in range(self.batch_size_att):
                        if not dones[i]:
                            self.success_memory.append(Transition(state[i], torch.tensor([actions[i]]), next_state[i], torch.tensor([-2]), True))

                if self.num % 3 == 0:
                    loss = self.optimize_model()
                    self.loss_lists.append(loss)
                if self.num % 25 == 0:
                    self.epsilon = max(0.1, self.epsilon - 0.001)
                if self.num % self.TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if self.num % 1000 == 0:
                    # torch.save(self.policy_net.state_dict(), r"model_0_trenvong_2.pth")
                    # with open(r"actions.json", "w") as f:
                    #     json.dump(self.action_lists, f)
                    with open(r"losses.txt", "w") as f:
                        json.dump(self.loss_lists, f)
                self.num += 1
                state = next_state

            print(f"Success: {sum(dones)}/{self.batch_size_att}")
            his.append({
                "success": sum(dones),
                "total": self.batch_size_att,
                'espilon': self.epsilon
            })
            
            with open(r"logs.txt", "a") as f:
                for i in range(self.batch_size_att):
                    continue
                    save_image(new_img[i], f"final/{k}_{i}.png")
                    f.write(f"{k}_{i}: {label[i]} -> {prob[i].argmax()} with {torch.norm(new_img[i] - img[i])}, prob_label: {orig_prob[i][label[i]].item()} -> {prob[i][label[i]].item()}, prob_max: {prob[i].max().item()}\n")
            # with open(r"logs.txt", "a") as f:
            #     f.write(f"{k}: {label} -> {prob.argmax()} with {torch.norm(img - orig_img)}, prob_label: {orig_prob[0][label].item()} -> {prob[0][label].item()}, prob_max: {prob.max().item()}\n")
        print("Finished training")
        torch.save(self.policy_net.state_dict(), r"model_0_trrenvong_2.pth")
        # with open(r"actions.json", "w") as f:
        #     json.dump(self.action_lists, f)
        # with open(r"rewards.json", "w") as f:
        #     json.dump(self.reward_lists, f)
        with open(r"losses.json", "w") as f:
            json.dump(self.loss_lists, f)
        with open(r"his.json", "w") as f:
            json.dump(his, f)

if __name__ == "__main__":

    if not os.path.exists("original"):
        os.mkdir("original")
    if not os.path.exists("final"):
        os.mkdir("final")

    agent = Agent_Attack(device="cuda")
    agent.train()

