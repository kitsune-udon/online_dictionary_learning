import mini_batch_dictionary_learning as dl
import numpy as np
import omp
import imageio

def learn(src_img, patch_size, n_samples, k, m):
    w, h = src_img.shape
    pw, ph = patch_size
    params = dl.initialize_params(pw*ph, k, m)
    for i in xrange(n_samples):
        print "learn ({}/{})".format(i+1, n_samples)
        xn, yn = w-pw+1, h-ph+1
        x = np.random.randint(xn)
        y = np.random.randint(yn)
        d = src_img[y:y+ph, x:x+pw].ravel()
        dl.learn(d, params)
    return dl.get_dictionary(params)

def encode(src_img, D, patch_size, m):
    h, w = src_img.shape
    pw, ph = patch_size
    k = D.shape[1]
    xn, yn = w/pw, h/ph
    C = np.zeros((yn, xn, k))
    for j in xrange(yn):
        for i in xrange(xn):
            print "encode ({}/{})".format(i+j*xn+1, xn*yn)
            x, y = i * pw, j * ph
            v = src_img[y:y+ph, x:x+pw].ravel()
            c = omp.omp(v, D, m)
            C[j,i] = c
    return C

def decode(C, D, patch_size, m):
    yn, xn, k = C.shape
    pw, ph = patch_size
    dst_img = np.zeros((yn*ph, xn*pw))
    for j in xrange(yn):
        for i in xrange(xn):
            print "decode ({}/{})".format(i+j*xn+1, xn*yn)
            c = C[j,i]
            v = np.dot(D, c)
            x, y = i * pw, j * ph
            dst_img[y:y+ph, x:x+pw] = v.reshape((ph, pw))
    return dst_img

patch_size = (8, 8)
n_samples = 2000
# k : num of atoms
k = 128
# m : sparsity param
m = 16

np.random.seed(0)
img = imageio.read_image_L('lena.png')
D = learn(img, patch_size, n_samples, k, m)
C = encode(img, D, patch_size, m)
decoded_img = decode(C, D, patch_size, m)
imageio.write_image_L("lena_out_m{}.png".format(m), decoded_img)
