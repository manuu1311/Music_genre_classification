import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

def make_analysis(y,yhat):
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y, yhat)
    confmat=metrics.confusion_matrix(y,yhat)
    accuracy=np.diagonal(confmat/np.sum(confmat,axis=1))
    class_labels = ['Blues','Classical','Country','Disco','HipHop','Jazz','Metal','Pop','Reggae','Rock']
    metrics_df = DataFrame({
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': fscore,
        'Support': support
    }, index=class_labels)
    metrics_df.loc['Average'] = metrics_df.mean()
    print(metrics_df)
    # Plotting
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # Precision
    ax[0, 0].bar(np.arange(len(class_labels)), precision, color='skyblue')
    ax[0, 0].set_title('Precision')
    ax[0, 0].set_xticks(np.arange(len(class_labels)))
    ax[0, 0].set_xticklabels(class_labels, rotation=45, ha='right')

    # Recall
    ax[0, 1].bar(np.arange(len(class_labels)), recall, color='salmon')
    ax[0, 1].set_title('Recall')
    ax[0, 1].set_xticks(np.arange(len(class_labels)))
    ax[0, 1].set_xticklabels(class_labels, rotation=45, ha='right')

    # F1 Score
    ax[1, 0].bar(np.arange(len(class_labels)), fscore, color='lightgreen')
    ax[1, 0].set_title('F1 Score')
    ax[1, 0].set_xticks(np.arange(len(class_labels)))
    ax[1, 0].set_xticklabels(class_labels, rotation=45, ha='right')

    # Support
    ax[1, 1].bar(np.arange(len(class_labels)), accuracy, color='gold')
    ax[1, 1].set_title('Accuracy')
    ax[1, 1].set_xticks(np.arange(len(class_labels)))
    ax[1, 1].set_xticklabels(class_labels, rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()
    #confusion matrix
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confmat)
    disp.plot()
    plt.show()

def hist_plot(loss,acc, loss_valid,acc_valid):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].plot(loss,label='training loss')
    ax[0].plot(loss_valid,label='validation loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()
    ax[1].plot(acc,label='training accuracy')
    ax[1].plot(acc_valid,label='validation accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    ax[1].set_ylim(ymin=0,ymax=100)
    plt.show()