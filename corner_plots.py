import pickle
import corner
import numpy as np
import matplotlib.pyplot as plt


def plot_corners(path, model, ground_truth, dict_data, true, labels=[r'logM$_{halo}$', r'R_s', r'$q$', r'$\hat{x}$', r'$\hat{y}$', r'$\hat{z}$', 
                                r'x$_0$', r'y$_0$', r'z$_0$', r'v', r'$\hat{v_x}$', r'$\hat{v_y}$', r'$\hat{v_z}$']):

        params_data = np.loadtxt(f'{path}/params.txt')
        with open(f'{path}/dict_result.pkl', 'rb') as file:
                dns = pickle.load(file)
        with open(f'{path}/dict_data.pkl', 'rb') as file:
                dict_data = pickle.load(file)

        
        figure = corner.corner(dns['samps'], 
                    labels=labels,
                    color='blue',
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, 
                    title_kwargs={"fontsize": 16},
                    truths=ground_truth, 
                    truth_color='red')
        figure.savefig(f'{path}/corner_plot.pdf')
        

        plt.figure(figsize=(15, 5))
        plt.subplot(1,2,1)
        plt.xlabel(r'x [kpc]')
        plt.ylabel(r'y [kpc]')
        xyz_model, xyz_prog = model(dns['samps'][np.argmax(dns['logl'])])
        x_model, y_model = xyz_model[:,0], xyz_model[:,1]
        plt.plot(x_model, y_model, color='red', label='Orbit Model')
        xyz_data, xyz_data_prog = model(params_data)
        x_data, y_data = xyz_data[:,0], xyz_data[:,1]
        plt.plot(x_data, y_data, color='k', label='Orbit Data')
        plt.scatter(dict_data['x'], dict_data['y'], s=100, color='k', label='Stream Data')
        plt.legend(loc='best')
        plt.subplot(1,2,2)
        r_model = np.sqrt(x_model**2 + y_model**2)
        theta_model = np.arctan2(y_model, x_model)
        theta_model[theta_model < 0] += 2*np.pi
        theta_model = np.unwrap(theta_model)
        plt.plot(theta_model, r_model, color='red', label='Orbit Model')
        r_data = np.sqrt(x_data**2 + y_data**2)
        theta_data = np.arctan2(y_data, x_data)
        theta_data[theta_data < 0] += 2*np.pi
        theta_data = np.unwrap(theta_data)
        plt.plot(theta_data, r_data, color='k', label='Orbit Data')
        plt.scatter(dict_data['theta'], dict_data['r'], s=100, color='k', label='Stream Data')
        plt.legend(loc='best')
        plt.savefig(f'{path}/best_fit.pdf')
    

def plot_corners_population(path_save, N, q_mean, q_sig):
    # Plot the corner plot
    samples = np.load(path_save+f'/population_samples_N{N}.npy')
    fig = corner.corner(samples, 
                        color='blue',
                        quantiles=[0.16, 0.5, 0.84], 
                        show_titles=True, 
                        title_kwargs={"fontsize": 16},
                        truths=[q_mean, q_sig], 
                        truth_color='red',
                        labels=[r"q$_{mean}$", r"q$_{sigma}$"])
    figure.savefig(f'{path}/population_corner_plot_N{N}.pdf')
    return fig