from .config import CFG


class LLM:
    def generate(
        self,
        prompt: str,
        temperature: float = CFG["general"]["temperature"],
        n_generations: int = 1,
        length_normalize: bool = CFG["general"]["length_normalize"],
        **kwargs
    ):
        """
        Generate text in response to a prompt.

        Parameters
        ----------
        prompt : str
            The prompt you want to send to the model.
        temperature : float
            Softmax temperature for sampling. 0 is fully greedy.
        n_generations : int
            The number of samples you want to draw from the LLM.
        length_normalize : bool
            Whether you want to length-normalize the log-probability of the sequence.

        *SHOULD* Return
        ---------------
        tuple : (text, log_prob)
            The response text(s) and corresponding sequence(s)' log-probability(ies).
        """
        raise NotImplementedError
