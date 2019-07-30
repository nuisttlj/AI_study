import MTCNNnet_new
import trainer
import os

if __name__ == '__main__':
    net = MTCNNnet_new.ONet()

    if not os.path.exists("./param_widerface_new/"):
        os.makedirs("./param_widerface_new/")

    _trainer = trainer.Trainer(net, './param_widerface_new/onet.pt', r"C:\mywiderface_train\48", r"C:\mywiderface_dev\48", 48,
                               [0.9, 0.99, 0.999, 0.9999], "Onet", isCuda=True)
    _trainer.train()

