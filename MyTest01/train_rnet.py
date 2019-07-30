import MTCNNnet_samestride
import trainer
import os

if __name__ == '__main__':
    net = MTCNNnet_samestride.RNet()

    if not os.path.exists("./param_widerface_new/"):
        os.makedirs("./param_widerface_new/")

    _trainer = trainer.Trainer(net, './param_widerface_new/rnet.pt', r"C:\mywiderface_train\24", r"C:\mywiderface_dev\24", 24,
                               [0.9, 0.99, 0.999], "Rnet", isCuda=True)
    _trainer.train()
