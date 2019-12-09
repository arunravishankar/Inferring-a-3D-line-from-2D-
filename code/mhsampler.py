import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os


DATA_ROOT = os.path.join('..', 'data')

FIGURES_ROOT = os.path.join('..', 'figures')

PATH_TO_xi = os.path.join(FIGURES_ROOT, 'p_i_x.png')
PATH_TO_yi = os.path.join(FIGURES_ROOT, 'p_i_y.png')
PATH_TO_zi = os.path.join(FIGURES_ROOT, 'p_i_z.png')

PATH_TO_xf = os.path.join(FIGURES_ROOT, 'p_f_x.png')
PATH_TO_yf = os.path.join(FIGURES_ROOT, 'p_f_y.png')
PATH_TO_zf = os.path.join(FIGURES_ROOT, 'p_f_z.png')


PATH_TO_xi12 = os.path.join(FIGURES_ROOT, 'p_i_x12.png')
PATH_TO_yi12 = os.path.join(FIGURES_ROOT, 'p_i_y12.png')
PATH_TO_zi12 = os.path.join(FIGURES_ROOT, 'p_i_z12.png')

PATH_TO_xf12 = os.path.join(FIGURES_ROOT, 'p_f_x12.png')
PATH_TO_yf12 = os.path.join(FIGURES_ROOT, 'p_f_y12.png')
PATH_TO_zf12 = os.path.join(FIGURES_ROOT, 'p_f_z12.png')

PATH_TO_cam1 = os.path.join(FIGURES_ROOT, 'cam1.png')
PATH_TO_cam2 = os.path.join(FIGURES_ROOT, 'cam2.png')
PATH_TO_cam12 = os.path.join(FIGURES_ROOT, 'cam12.png')
PATH_TO_cam21 = os.path.join(FIGURES_ROOT, 'cam21.png')

t = np.genfromtxt('../data/inputs.csv', dtype=None)
points1 = np.genfromtxt('../data/points_2d_camera_1.csv', dtype=None, delimiter=',')
points2 = np.genfromtxt('../data/points_2d_camera_2.csv', dtype=None, delimiter=',')

camera_pos1 = np.zeros((3, 1))
camera_pos2 = np.transpose([[-5, 0, 5]])

I = np.identity(3)
M = np.concatenate((I, camera_pos1), axis=1)  # camera matrix 1


I_prime = np.fliplr(np.diag([1, 1, -1]))
M_prime = np.concatenate((I_prime, camera_pos2), axis=1)  # camera matrix 2


mu_l = np.array([0, 0, 4])  # 3D line parameters
sigma_l = np.dot(6, I)

cov1 = np.cov(points1.T)  # covariance matrix for rs distribution
cov2 = np.cov(points2.T)

si = np.dot(0.05**2, np.identity(3))


n = 1000000  # iterations


def start_point():
    """
    First sample of random walk, sampling from prior
    :return start: Returns (p_i, p_f) which is a set of two 3 dimensional points sampled from a multivariate Gaussian distribution
    """
    start = np.random.multivariate_normal(mu_l, sigma_l, (1, 2))

    return start


def convert_to_2d(p, M):
    """
    Convert 3D point to 2D point
    :param p: Point in 3 dimensions to be converted to a 2 dimensional point
    :param M: Camera Matrix M 3x4 Matrix
    :return q: 2 dimensional point
    """
    step_1 = np.matrix(np.insert(p, 3, 1)).T  # insert 1 to 3D point
    u_v_w = M * step_1  # multiply new 3D point with camera matrix M
    u_v = np.concatenate((u_v_w[0], u_v_w[1]))
    q = np.divide(u_v, u_v_w[2])  # 2D point
    return q


def proposal(p,si):
    """
    Proposal distribution; proposing a point based on the previous point
    :param p: p is a 3 dimensional point
    :return prop: Returns a point based by sampling from a Gaussian distribution based on the previous point as the mean
    """
    prop = np.random.multivariate_normal(p, si)

    return prop


def log_prior(pi, pf):
    """
    Log Scale of Prior
    :param pi: Initial point pi in 3 dimensions
    :param pf: Final point pf in 3 dimensions
    :return y_i+y_f: returns the log of the prior probability that the point is (p_i,p_f)
    """
    y_i = multivariate_normal.logpdf(pi, mu_l, sigma_l)
    y_f = multivariate_normal.logpdf(pf, mu_l, sigma_l)

    return y_i + y_f


def data_likelihood(pi, pf, r, M, cov):
    """
    Log Scale of Likelihood
    :param pi: Initial point pi in 3 dimensions
    :param pf: Final point pf in 3 dimensions
    :param r: Given points r_s
    :param M: Camera Matrix M
    :param cov: Covariance of the data points given
    """
    log_likelihood = 0

    qi = convert_to_2d(pi, M)  # convert 3D points to 2D
    qf = convert_to_2d(pf, M)

    for t_s, r_s in zip(t, r):  # read 2D point r_s and input t_s
        q_s = qi + np.dot(qf - qi, t_s) #Sampling the points in 2 dimensions
        mu = np.array(q_s).flatten() 
        y_s = multivariate_normal.logpdf(r_s, mu, cov)
        log_likelihood += y_s
    return log_likelihood


def calculate_posterior(pi, pf, r, M, cov, pi_prime = None, pf_prime = None, r_prime = None, M_prime = None, cov_prime = None):
    """
    Log Scale of Posterior
    :param pi: Initial point pi in 3 dimensions
    :param pf: Final point pf in 3 dimensions
    :param r: Given points r_s
    :param M: Camera Matrix M
    :param cov: Covariance of data points given
    :return : Log of the Posterior
    """
    primed=0
    if pi_prime is not None:
        if pf_prime is not None:
            primed = data_likelihood(pi_prime, pf_prime, r_prime, M_prime, cov_prime)
    log_posterior = log_prior(pi, pf) + data_likelihood(pi, pf, r, M, cov) + primed
    return log_posterior


def metropolis_hastings(r, M, cov, si, r_prime=None, M_prime=None, cov_prime=None):
    """
    Metropolis Hastings Algorithm
    :param r: Given points r_s
    :param M: Camera matrix M
    :param cov: Covariance of data points given
    :return samples, acc_rate:
    """

    # sample the start 3D points p_i, p_f
    pi_star, pf_star = start_point()[:,0].flatten(), start_point()[:,1].flatten()
    pi_star_prime, pf_star_prime = start_point()[:,0].flatten(), start_point()[:,1].flatten()
    
    samples = np.zeros((n+1, 6))  # initialize sampler
    samples[0][:3] = pi_star
    samples[0][3:] = pf_star

    cur_log_prob = calculate_posterior(pi_star, pf_star, r, M, cov)  # current log probability of posterior

    if r_prime is not None:
       cur_log_prob = calculate_posterior(pi_star, pf_star, r, M, cov, pi_star_prime, pf_star_prime, r_prime, M_prime, cov_prime) #If a second camera is included, add the log_prob of the new camera to this

    acc_count = 0  # sample acceptance rate

    for i in range(1, n+1):

        #if not i % 500:
        #    print('Iteration %i' % i)

        new_pi = proposal(samples[i-1][:3],si)  # propose new samples
        new_pf = proposal(samples[i-1][3:],si)
        new_log_prob = calculate_posterior(new_pi, new_pf, r, M, cov)  # new log probability of posterior

        ratio = new_log_prob - cur_log_prob  # acceptance ratio
        alpha = min(np.log(1), ratio)

        u = np.random.rand()

        if np.log(u) <= alpha:
            # accept new samples
            samples[i][:3] = new_pi
            samples[i][3:] = new_pf
            cur_log_prob = new_log_prob
            acc_count += 1
        else:
            # reject new samples
            samples[i] = samples[i-1]

    acc_rate = acc_count / float(n)

    return samples, acc_rate


def monte_2d(pi, pf, t_pre, cov):
    """
    Monte Carlo estimate of 2D output point
    :param pi: Initial point pi in 3 dimensions
    :param pf: Final point pf in 3 dimensions
    :param t_pre: Parameterizing the points qs in 2 dimensions
    :param cov: Covariance of the data
    :return rs: Returns 2 dimensional points rs
    """
    qi = convert_to_2d(pi, M)
    qf = convert_to_2d(pf, M)
    qs = qi + np.dot(qf - qi, t_pre)
    rs = np.random.multivariate_normal(np.array(qs).flatten(), cov)

    return rs

def plot_mh_sampling(sample,plot_number,save_fig,plot_title):
    fig = plt.figure()
    b = sample
    if plot_number == 0:
        name = r'$P_i (x)$'
    elif plot_number == 1:
        name = r'$P_i (y)$'
    elif plot_number == 2:
        name = r'$P_i (z)$'
    elif plot_number == 3:
        name = r'$P_f (x)$'
    elif plot_number == 4:
        name = r'$P_f (y)$'
    elif plot_number == 5:
        name = r'$P_f (z)$'

    fig.suptitle(plot_title)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(n+1), b)
    plt.xlabel('Number of Samples')
    plt.ylabel(r'Value of sampled %s' %name)

    plt.subplots_adjust(hspace=0.5)
    
    plt.subplot(2, 1, 2)
    plt.hist(b, bins=80)
    plt.xlabel(r'Value of sampled %s' %name)
    plt.ylabel('Number of Observations')
    plt.savefig(save_fig, fmt = 'png')
    plt.show()
    
    return 


def plot_camera(point,M,plot_title,save_fig):    
    
    q_xi = np.array(convert_to_2d(point[:3],M)[0])
    q_xf = np.array(convert_to_2d(point[3:],M)[0])
    q_yi = np.array(convert_to_2d(point[:3],M)[1])
    q_yf = np.array(convert_to_2d(point[3:],M)[1])
    q_x = np.array([q_xi[0][0], q_xf[0][0]]) 
    q_y = np.array([q_yi[0][0], q_yf[0][0]])
    fig = plt.figure()
    fig.suptitle(plot_title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(q_x,q_y)
    plt.scatter(points1[:,0],points1[:,1])
    plt.savefig(save_fig, fmt='png')
    plt.show()
    
    return

# -------------
# MAIN FUNCTION
# -------------

mh_sample1, accu1 = metropolis_hastings(points1, M, cov1,si)
mh_sample2, accu2 = metropolis_hastings(points2, M_prime, cov2,si)

mh_sample3, accu3 = metropolis_hastings(points1, M, cov1, si, points2, M_prime, cov2)



#print('\nCalculating MAP and Monte Carlo estimate...\n')
P1 = []
P2 = []
P3 = []
for i in range(1, n+1):
    prob1 = calculate_posterior(mh_sample1[i][:3], mh_sample1[i][3:], points1, M, cov1)
    prob2 = calculate_posterior(mh_sample2[i][:3], mh_sample2[i][3:], points2, M_prime, cov2)
    prob3 = calculate_posterior(mh_sample3[i][:3], mh_sample3[i][3:], points1, M, cov1, mh_sample3[i][:3], mh_sample3[i][3:],points2, M_prime, cov2)
    P1.append(prob1)
    P2.append(prob2)
    P3.append(prob3)

max_idx1, max_value1 = max(enumerate(set(P1)))
MAP1 = mh_sample1[max_idx1]
l1 = euclidean(MAP1[:3], MAP1[3:])

# Task 2
Path_to_filename = np.array([PATH_TO_xi,PATH_TO_yi,PATH_TO_zi,PATH_TO_xf,PATH_TO_yf,PATH_TO_zf])

plot_titlemh = 'MH sampler for inferring 3D points from 2D Images (camera 1)'
for i in range(0,6):
    plot_mh_sampling(mh_sample1[:,i], i, Path_to_filename[i], plot_titlemh)

# Task 3
max_idx1, max_value1 = max(enumerate(set(P1)))
MAP1 = mh_sample1[max_idx1]
l1 = euclidean(MAP1[:3], MAP1[3:])
print('\nThe MAP estimate for the initial point is %s\n' % MAP1[:3])
print('\nThe MAP estimate for the final point is %s\n' % MAP1[3:])
print('\nThe MAP estimate of 3D line length (euclidean) is %s\n' % l1)

plot_title2d = '2D projection of MAP estimate (camera 1)'
plot_camera(MAP1, M, plot_title2d, PATH_TO_cam1)



# Task 4
max_idx2, max_value2 = max(enumerate(set(P2)))
MAP2 = mh_sample2[max_idx2]
l2 = euclidean(MAP2[:3], MAP2[3:])
print('\nThe MAP estimate for the initial point is %s\n' % MAP2[:3])
print('\nThe MAP estimate for the final point is %s\n' % MAP2[3:])
print('\nThe MAP estimate of 3D line length (euclidean) is %s\n' % l2)


plot_title2d2 = '2D projection of MAP estimate (camera 2)'
plot_camera(MAP2, M_prime, plot_title2d2, PATH_TO_cam2)



# Task 5
Path_to_filename12 = np.array([PATH_TO_xi12,PATH_TO_yi12,PATH_TO_zi12,PATH_TO_xf12,PATH_TO_yf12,PATH_TO_zf12])
plot_titlemh2 = 'MH sampler for inferring 3D points from 2D Images (camera 2)'
for i in range(0,6):
    plot_mh_sampling(mh_sample3[:,i], i, Path_to_filename12[i], plot_titlemh2)

plot_title2d2c1 = '2D projection of MAP estimate (camera 1) - likelihood from both cameras'
plot_title2d2c2 = '2D projection of MAP estimate (camera 2) - likelihood from both cameras'
max_idx3, max_value3 = max(enumerate(set(P3)))
MAP3 = mh_sample1[max_idx3]
l3 = euclidean(MAP3[:3], MAP3[3:])
print('\nThe MAP estimate for the initial point is %s\n' % MAP3[:3])
print('\nThe MAP estimate for the final point is %s\n' % MAP3[3:])
print('\nThe MAP estimate of 3D line length (euclidean) is %s\n' % l3)
plot_camera(MAP3, M, plot_title2d2c1, PATH_TO_cam12)
plot_camera(MAP3, M_prime, plot_title2d2c2, PATH_TO_cam21)


