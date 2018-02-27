import pickle
import pdb

with open('dump.pkl', 'rb') as file:
    imgs, labels, reg_loss_list, cls_loss_list, oal_loss_list, preds, classifications, gtt, slt = pickle.load(file)

# plot cost development
import matplotlib.pyplot as plt
plt.figure()
plt.plot(reg_loss_list, label='reg loss')
plt.plot(cls_loss_list, label='cls loss')
plt.plot(oal_loss_list, label='overall loss')
plt.legend()
plt.show()

pdb.set_trace()


for i in range(1):
    plt.imshow(imgs[i, :, :]); plt.show()
    print(labels[i])
    print('predictions tensor:')
    print('x:',preds[0,:,:,0])
    print('y:',preds[0, :, :, 9])
    print('w:',preds[0, :, :, 18])
    print('h:',preds[0, :, :, 27])
    print('ground truth tensor tensor:')
    print('x:', gtt[0, :, :, 0])
    print('y:', gtt[0, :, :, 9])
    print('w:', gtt[0, :, :, 18])
    print('h:', gtt[0, :, :, 27])
    # results in a tensor with shape (1, IMG_SIZE, IMG_SIZE, 36)

    import pdb

    pdb.set_trace()

