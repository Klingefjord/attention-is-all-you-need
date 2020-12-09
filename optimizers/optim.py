class NoamOptimizer():
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._rate = 0
        self._step = 0
        
    def step(self):
        self._step += 1
        
        for p in self.optimizer.param_groups:
            p['lr'] = self.learning_rate(self._step)
            
        self.optimizer.step()
            
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def learning_rate(self, step_num):
        return self.d_model**-0.5 * min(step_num**-0.5,step_num*self.warmup_steps**-1.5)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    opts = [NoamOptimizer(None, 512, 4000),NoamOptimizer(None, 512, 2000), NoamOptimizer(None, 256, 4000), NoamOptimizer(None, 512, 10000),NoamOptimizer(None, 1024, 400)]
    plt.plot(np.arange(1, 20000), [[opt.learning_rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:2000", "256:4000", "512:10000", "512:400"])