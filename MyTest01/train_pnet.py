import MTCNNnet
import trainer
import os

if __name__ == '__main__':
    net = MTCNNnet.PNet()

    if not os.path.exists("./param_widerface_new/"):
        os.makedirs("./param_widerface_new/")

    _trainer = trainer.Trainer(net, './param_widerface_new/pnet.pt', r"C:\mywiderface_train\12", r"C:\mywiderface_dev\12",
                               12,
                               [0.8, 0.9, 0.99], "Pnet", isCuda=True)
    _trainer.train()
