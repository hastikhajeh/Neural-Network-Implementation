import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def initial_visualization(d2_X,d2_Y,d5_X,d5_Y):
    X_1 = d2_X[d2_Y==1]
    X_0 = d2_X[d2_Y==0]
    plt.subplot(1, 2, 1)
    plt.title("2d data")
    plt.plot(X_1[:,0],X_1[:,1],'ro',color='red')
    plt.plot(X_0[:,0],X_0[:,1],'ro',color='blue')
    X_1 = d5_X[d5_Y==1]
    X_0 = d5_X[d5_Y==0]
    plt.subplot(1, 2, 2)
    plt.title("5d data")
    plt.plot(X_1[:,0],X_1[:,1],'ro',color='red')
    plt.plot(X_0[:,0],X_0[:,1],'ro',color='blue')
    plt.show()

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def phi(W, b, X):
    return sigmoid((W.T @ X) + b)

def cross_entropy(W, b, X, y):
    c_phi = phi(W, b, X)
    return  -((y * np.log(c_phi)) + ((1 - y) * np.log(1 - c_phi)))

def dC_dW(W ,b ,X ,y):
    c_phi = phi(W, b, X)
    return (c_phi - y) * X

def dC_db(W, b, X, y):
    c_phi = phi(W, b, X)
    return (c_phi - y)

def initialize_weights(size):   
    return np.random.rand(size,1),np.random.rand()

def dC_dW_n(W, b, X, y):
    g = np.zeros(X.shape)
    dlt = 1e-7
    cost = cross_entropy(W, b, X, y)

    for i in range(g.shape[0]):
        W_c = W.copy()
        W_c[i, :] += dlt
        cost2 = cross_entropy(W_c, b, X, y)
        g[i, :] = (cost2 - cost) / dlt

    return g

def dC_db_n(W, b, X, y):
    g = np.zeros(X.shape[1])
    dlt = 1e-7
    cost = cross_entropy(W, b, X, y)
    b_c = b
    b_c += dlt
    cost2 = cross_entropy(W, b_c, X, y)        
    g = (cost2 - cost) / dlt    
    return g

def check_gradient(W, b, X, y):
    d_dw = dC_dW(W,b,X,y)
    d_db = dC_db(W,b,X,y)
    d_dw_n = dC_dW_n(W,b,X,y)
    d_db_n = dC_db_n(W,b,X,y)
    rel_err_w = np.linalg.norm(d_dw-d_dw_n)/np.linalg.norm(d_dw)
    rel_err_b = np.linalg.norm(d_db-d_db_n)/np.linalg.norm(d_db)
    return rel_err_w,rel_err_b

def mean_gradient(w,b,X,Y):
    dw = dC_dW(w,b,X,Y)
    db = dC_db(w,b,X,Y)
    dwm = np.sum(dw,axis=1)/X.shape[1]
    dwm = dwm.reshape((X.shape[0],1))
    dbm = np.sum(db,axis=1)/X.shape[1]
    dbm = dbm.reshape((1,1))
    return dwm,dbm

def err(y_pred,Y):
    err = np.zeros((y_pred[0].shape))
    # print(y_pred[0].shape)
    for i,y in enumerate(y_pred[0]):
        if y >= 0.5:
            if(Y[0][i] == 0):
                err[i] = 1
        else:
            if(Y[0][i] == 1):
                err[i] = 1
    final_err = np.sum(err)/(y_pred[0].shape[0])
    return final_err,err

def gradient_decent(lamb,w,b,X,Y):
    # lamb = 0.05
    dw,db = mean_gradient(w,b,X,Y)
    ws = []
    bs = []
    es = []
    while True:
        w -= lamb * dw
        b -= lamb * db
        if np.linalg.norm(dC_dW(w,b,X,Y)) < 0.8 and np.linalg.norm(dC_db(w,b,X,Y)) < 0.7:
            break
        cost = np.sum(cross_entropy(w,b,X,Y))
        y_pred = phi(w,b,X)
        final_error,e = err(y_pred,Y)
        print(final_error)
        print(cost)
        ws.append(w.tolist())
        bs.append(b.tolist())
        es.append(e.tolist())
    return ws,bs,es

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals[0], '--')

def visualize(i,X,y,w,b,e, ax):
    w = np.array(w)
    b = np.array(b)
    e = np.array(e)
    # print('------------------------')
    X = X.T
    y = y.flatten()
    ec = e[i].flatten()
    X_1 = X[y==1]
    X_0 = X[y==0]
    X_e = X[ec==1]
    
    plt.title("2d data")
    ax.clear()
    ax.plot(X_1[:,0],X_1[:,1],'ro',color='red')
    ax.plot(X_0[:,0],X_0[:,1],'ro',color='blue')
    ax.plot(X_e[:,0],X_e[:,1],'ro', color = 'white')
    ax.plot(X_e[:,0],X_e[:,1],'o',mfc = 'none', color = 'black')
    abline(-w[i][0]/w[i][1],-b[i]/w[i][1])
    ax.set_xlim([-5,5])
    ax.set_ylim([-7,7])
    # plt.show()


raw_data_2d = np.load('data2d.npz')
X2 = raw_data_2d['X'].T
# X2 = X2[:,:-1]
Y2 = raw_data_2d['y']
# Y2 = Y2[:-1]
raw_data_5d = np.load('data5d.npz')
X5 = raw_data_5d['X'].T
Y5 = raw_data_5d['y']
Y2 = Y2.reshape((70,1)).T
Y5 = Y5.reshape((70,1)).T
# X2 -> (2,70) Y2 -> (1,70) w2 -> (2,1) b2 -> ()
# X2 -> (5,70) Y2 -> (1,70) w2 -> (5,1) b2 -> ()
w2,b2 = initialize_weights(X2.shape[0])
w5,b5 = initialize_weights(X5.shape[0])
w,b,e = gradient_decent(0.1,w2,b2,X2,Y2)
fig, ax = plt.subplots(1,1)
w1 = np.array(w)
ani = FuncAnimation(fig, visualize, fargs=(X2,Y2,w,b,e, ax), frames=w1.shape[0], interval=500, repeat=False)
plt.show()
# plt.close()
