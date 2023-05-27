    def predict(self, input: cv2.Mat, w: np.ndarray = None) -> None | np.ndarray:
        w = w or self.weights

        if len(w) != len(self.experts):
            return None

        input_id = ray.put(input)
        ray_ids = [apply_expert.remote(input_id, model) for model in self.experts]
        r = ray.get(ray_ids)

        wbboxes = [
            [w[i], np.array(b), np.float32(c)]
            for i, p in enumerate(r)
            for b, c in zip(*p)
        ]

        return deduplicate_wbboxes(wbboxes)
