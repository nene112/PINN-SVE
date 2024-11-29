import math
import torch
import numpy as np
from network import Network

class PINN_SaintVenant:
    def __init__(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = Network(
            input_size=2,  # x and t
            hidden_size=16,
            output_size=2,  # h (water depth) and u (discharge)
            depth=8,
            act=torch.nn.Tanh
        ).to(device)

        self.dx = 0.1
        self.dt = 0.1
        # 21
        x = torch.arange(-1, 1 + self.dx, self.dx)  # 在[-1,1]区间上均匀取值，记为x
        # 11
        t = torch.arange(0, 1 + self.dt, self.dt)  # 在[0,1]区间上均匀取值，记为t
        #print(x.shape)
        #print(t.shape)
        # 整个时空计算域内的坐标(x,t)mesh，包含初边界mesh
        ##将x和t组合，形成时间空间网格，记录在张量X_inside中,
        # 21*11=231,(231,2),代表231行，每一行是一个长度为2的数组
        self.X_inside = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T
        #print(self.X_inside.shape)
        #print(self.X_inside)

        # 边界处的时空坐标 mesh
        ## 只是生成了这些点的坐标，它本身并没有赋值任何物理量或变量值
        ## .reshape(2, -1) 会将3维张量重塑为2维张量。这里的 -1 表示让PyTorch自动计算这一维度的大小。
        bc1 = torch.stack(torch.meshgrid(x[0], t)).reshape(2, -1).T  # x=-1边界,左边界x[0]在所有时间点的值都等于预设值
        bc2 = torch.stack(torch.meshgrid(x[-1], t)).reshape(2, -1).T  # x=+1边界，右边界所有时间都等于预设值
        ic = torch.stack(torch.meshgrid(x, t[0])).reshape(2, -1).T  # t=0边界，初始时间所有点都等于预设x
        ic1 = torch.stack(torch.meshgrid(x, t[0])).reshape(2, -1).T  # t=0边界，初始时间所有点都等于预设x
        self.X_boundary = torch.cat([bc1, bc2, ic,ic1])  # 将所有边界处的时空坐标点整合为一个张量
        #print("ic.shape:" ,ic.shape) # 21 * 2
        #print("len(bc1):" ,len(bc1)) # 11
        #print("len(ic):", len(ic))  # 21
        #print("self.X_boundary.shape:", self.X_boundary.shape)  # 64*2

        # 边界处的u值
        ## 要完成物理问题的建模，在接下来的代码中为这些坐标点分配相应的边界条件值
        # 边界处的h和u值
        u_bc1 = torch.zeros(len(bc1))  # 在左边界 x = -1 流速为 0
        u_bc2 = torch.zeros(len(bc2))  # 在右边界 x = 1 流速为 0
        u_ic = torch.zeros(len(ic))  # 初始流速 u = 0
        h_ic = torch.where(ic[:, 0] < 0, torch.ones(len(ic)),torch.ones(len(ic)) * 0.1)  # h = 1 for x < 0, h = 0.1 for x >= 0

        self.U_boundary = torch.cat([u_bc1, u_bc2,u_ic,h_ic])  # 将所有边界处的u值整合为一个张量# 11+11+21+21 = 64
        ## 假设 self.U_boundary 原本是一个形状为 (N,) 的一维张量，那么执行 unsqueeze(1) 后，
        ## 它的形状将变为 (N, 1)，即一个二维张量，其中第二个维度的大小为 1。
        self.U_boundary = self.U_boundary.unsqueeze(1)  # 64*1

        self.X_inside = self.X_inside.to(device)
        self.X_boundary = self.X_boundary.to(device)
        ## 初边值边界 赋值 对应着已有的mesh
        self.U_boundary = self.U_boundary.to(device)



        self.X_inside.requires_grad = True

        self.criterion = torch.nn.MSELoss()
        self.iter = 1

        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.adam = torch.optim.Adam(self.model.parameters())

        self.g = 9.81  # Gravity constant

    def loss_func(self):
        # 将导数清零
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        U_pred_boundary = self.model(self.X_boundary)
        #print(U_pred_boundary.shape)
        loss_boundary = self.criterion(U_pred_boundary, self.U_boundary)

        # 内部点的预测结果 (h, u)
        U_inside = self.model(self.X_inside)
        h_inside = U_inside[:, 0]  # 提取h
        u_inside = U_inside[:, 1]  # 提取u

        # #print("U_inside:", U_inside.shape)
        dh_dX = torch.autograd.grad(
            inputs=self.X_inside,# 对这个求导
            outputs=h_inside, #
            # 要指定
            grad_outputs=torch.ones_like(h_inside),  # 梯度的传播不考虑权重
            retain_graph=True,
            create_graph=True
        )[0]
        dh_dx = dh_dX[:, 0]  # 提取对第x的导数
        dh_dt = dh_dX[:, 1]  # 提取对第t的导数

        du_dX = torch.autograd.grad(
            inputs=self.X_inside,
            outputs=u_inside,
            # 要指定
            grad_outputs=torch.ones_like(u_inside),
            retain_graph=True,
            create_graph=True
        )[0]
        du_dx = du_dX[:, 0]  # 提取对第x的导数
        du_dt = du_dX[:, 1]  # 提取对第t的导数
        # #print("u_inside:", u_inside.shape)

        # 质量守恒方程残差
        loss_mass_conservation = self.criterion(dh_dt +u_inside* dh_dx, torch.zeros_like(h_inside))
        # 动量守恒方程残差
        Sf=((0.025**2)*abs(u_inside)*u_inside)/(1*h_inside/(2*h_inside + 1))**(4/3)
        S = 0.00001
        loss_momentum_conservation = self.criterion(
            du_dt + u_inside * du_dx + self.g * dh_dx + self.g *(Sf-S),  #
            torch.zeros_like(u_inside)
        )

        # 物理方程的损失
        loss_equation = loss_mass_conservation + loss_momentum_conservation
        loss = loss_equation + loss_boundary

        loss.backward()

        if self.iter % 100 == 0:
            print(f"Iteration {self.iter}, Loss: {loss.item()}")
        self.iter += 1
        return loss

    def Sf(self, h, q):
        n = 0.03  # Manning's roughness coefficient
        return (n ** 2 * q.abs() * q) / (h ** (10 / 3))

    def S0(self):
        return 0.001  # Channel bed slope

    def d_dx(self, u):
        return torch.autograd.grad(
            u, self.X_inside,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0][:, 0]

    def train(self):
        self.model.train()
        #print("Optimizing with Adam")
        # Adam：
        #
        # 是一种基于一阶梯度的自适应优化方法。
        # 适用于处理大规模问题，收敛速度快，对初始学习率不敏感。
        # 优点：无需额外的计算二阶导数信息，能快速收敛。
        # 缺点：对于某些问题可能收敛到次优解。
        # L - BFGS：
        #
        # 是一种基于二阶信息的优化方法（准牛顿法）。
        # 优点：收敛性好，适合处理小规模问题。
        # 缺点：需要更多计算资源，不适合大规模深度学习问题。
        #
        # Adam的学习率：确保Adam的学习率适当（如0.001），避免过大的梯度更新。
        # L - BFGS
        # 的设置：L - BFGS在PyTorch中需要一个闭包（closure）来重复调用损失函数，如果self.loss_func设计不当，可能会引发错误。
        # 混合优化策略：这是一种常见策略，通过不同优化器的组合来提升收敛效果。
        for _ in range(500):
            self.adam.step(self.loss_func)
        #print("Optimizing with L-BFGS")
        self.lbfgs.step(self.loss_func)

# Instantiate and train the model
pinn = PINN_SaintVenant()
pinn.train()

# Save the model
torch.save(pinn.model, 'saint_venant_model_dambreak.pth')
