
import matplotlib.pyplot as plt
import seaborn as sns


# Function to plot predictions by GPR model
def plot_pred(gpr, X):
    y_pred, y_std = gpr.predict(X, return_std=True)

    data_df = pd.DataFrame(
        {
            't':        X.reshape(sample_size).tolist()[0],
            'y':        y,
            'y_pred':   y_pred,
            'y_std':    y_std,
        }
    )

    data_df['y_pred_lwr'] = data_df['y_pred'] - 2 * data_df['y_std']
    data_df['y_pred_upr'] = data_df['y_pred'] + 2 * data_df['y_std']

    fig, ax = plt.subplots()

    ax.fill_between(
        x=X.reshape(sample_size).tolist()[0], 
        y1=data_df['y_pred_lwr'], 
        y2=data_df['y_pred_upr'], 
        color='grey', 
        alpha=0.15, 
        label='credible_interval'
    )

    sns.lineplot(x='t', y='y', data=data_df, color='red', label = 'y1', ax=ax)
    sns.lineplot(x='t', y='y_pred', data=data_df, color='blue', label='y_pred', ax=ax)    



# Function to plot sample from GPR model
def plot_sample(gp, n, X, y):
    gp_sample = gp.sample_y(X=X, n_samples=100)

    fig, ax = plt.subplots()
    for i in range(gp_sample.shape[1]):
        plt.plot(X, gp_sample[:,i], color='orange')
    plt.plot(X, y, 'o', color='grey')