import numpy as np

class BudgetAllocator():
    def __init__(self, budget, cfg, if_second=False):
        self.budget = budget
        self.max_epochs = cfg.TRAINER.MAX_EPOCHS
        self.cfg = cfg
        self.if_second = if_second
        self.build_budgets()

    def build_budgets(self):
        self.budgets = np.zeros(self.cfg.TRAINER.MAX_EPOCHS, dtype=np.int32)
        self.second_budgets = np.zeros(self.cfg.TRAINER.MAX_EPOCHS, dtype=np.int32)
        rounds = self.cfg.ADA.ROUNDS or np.arange(0, self.cfg.TRAINER.MAX_EPOCHS, self.cfg.TRAINER.MAX_EPOCHS // self.cfg.ADA.ROUND)
        rounds_second = self.cfg.SECOND.ROUNDS_2

        for r in rounds:
            if self.if_second and r in rounds_second:
                budget_epoch = self.budget // len(rounds)
                self.second_budgets[r] = np.round(budget_epoch*self.cfg.SECOND.BUDGET_2)
                self.budgets[r] = budget_epoch - self.second_budgets[r]
            else:
                self.budgets[r] = self.budget // len(rounds)

        self.budgets[rounds[-1]] += self.budget - self.budgets.sum() - self.second_budgets.sum()
        # print(self.budgets)
        # print(self.second_budgets)

    def get_budget(self, epoch):
        curr_budget = self.budgets[epoch]
        used_budget = self.budgets[:epoch].sum()
        if self.if_second:
            used_budget_second = self.second_budgets[:epoch].sum()
            curr_budget_second = self.second_budgets[epoch]
            return curr_budget, curr_budget_second, used_budget, used_budget_second
        else:
            return curr_budget, used_budget
