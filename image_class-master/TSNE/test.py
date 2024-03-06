import TSNE
import numpy as np

from TSNE.TSNE import plot_embedding

combined_test_labels = np.vstack([source_label[:250], target_label[:250]])
combined_test_domain = np.vstack([np.tile([1., 0.], [250, 1]), np.tile([0., 1.], [250, 1])])

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
dann_tsne_middle = tsne.fit_transform(test_emb_middle)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
dann_tsne_last = tsne.fit_transform(test_emb_last)

plot_embedding(dann_tsne_middle, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'middle')
plot_embedding(dann_tsne_last, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'last')