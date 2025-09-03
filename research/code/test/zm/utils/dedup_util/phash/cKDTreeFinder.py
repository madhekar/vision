from scipy.spatial import cKDTree
from utils.dedup_util.phash import NearDuplicateImageFinder


class cKDTreeFinder(NearDuplicateImageFinder.NearDuplicateImageFinder):
    valid_metrics = ["manhattan", "euclidean"]

    def __init__(
        self,
        img_file_list,
        distance_metric='manhattan',
        leaf_size=40,
        parallel=False,
        batch_size=32,
        verbose=0,
    ):
        self.distance_metric = distance_metric
        super().__init__(img_file_list, leaf_size, parallel, batch_size, verbose)

    def build_tree(self):
        print("Building the cKDTree...")
        assert self.distance_metric in self.valid_metrics, (
            "{} isn't a valid metric for cKDTree.".format(self.distance_metric)
        )

        hash_str_len = len(self.df_dataset.at[0, "hash_list"])
        self.tree = cKDTree(
            self.df_dataset[[str(i) for i in range(0, hash_str_len)]],
            leafsize=self.leaf_size,
        )

    def _find_all(self, nearest_neighbors=5, threshold=5):
        n_jobs = 1
        hash_str_len = len(self.df_dataset.at[0, "hash_list"])
        """
        p : float, 1<=p<=infinity
                   Which Minkowski p-norm to use. 
                   1 is the sum-of-absolute-values "Manhattan" distance
                   2 is the usual Euclidean distance
                   infinity is the maximum-coordinate-difference distance
        """
        p = 1  # default Manhattan
        if self.distance_metric == self.valid_metrics[0]:
            p = 1
        elif self.distance_metric == self.valid_metrics[1]:
            p = 2
        # 'distances' is a matrix NxM where N is the number of images and M is the value of nearest_neighbors_in.
        # For each image it contains an array containing the distances of k-nearest neighbors.
        # 'indices' is a matrix NxM where N is the number of images and M is the value of nearest_neighbors_in.
        # For each image it contains an array containing the indices of k-nearest neighbors.
        if self.parallel:
            print("\tCPU: {}".format(self.number_of_cpu))
            n_jobs = self.number_of_cpu

        distances, indices = self.tree.query(
            self.df_dataset[[str(i) for i in range(0, hash_str_len)]],
            k=nearest_neighbors,
            p=p,
            distance_upper_bound=threshold,
            n_jobs=n_jobs,
        )

        return distances, indices

    def _find(self, image_id, nearest_neighbors=5, threshold=10):
        n_jobs = 1
        if self.parallel:
            n_jobs = self.number_of_cpu

        hash_str_len = len(self.df_dataset.at[0, "hash_list"])

        distances, indices = self.tree.query(
            self.df_dataset[[str(i) for i in range(0, hash_str_len)]]
            .iloc[image_id]
            .values.reshape(1, -1),
            k=nearest_neighbors,
            p=1,
            distance_upper_bound=threshold,
            n_jobs=n_jobs,
        )

        return distances, indices
