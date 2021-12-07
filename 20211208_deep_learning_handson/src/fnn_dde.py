"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde


data = dde.data.DataSet(
    fname_train="train.txt", fname_test="test.txt", col_x=(0,), col_y=(1,)
)
net = dde.maps.FNN([1] + [128] * 4 + [1], "relu", "Glorot normal")

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=10000)

dde.saveplot(losshistory, train_state, issave=False, isplot=True)
