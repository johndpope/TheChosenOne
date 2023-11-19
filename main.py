import scipy as sp
from diffusers import StableDiffusionPipeline, PNDMScheduler
import numpy as np

# from diffusers import AutoencoderKL, UNet2DConditionModel, , DDIMScheduler

def consistent_character_generation(target_prompt, hyper_parameters, model_name):
    """
    Generates a consistent representation of the target prompt using the specified model.

    Args:
        target_prompt: The target prompt.
        hyper_parameters: A dictionary of hyper-parameters, including:
            * number_of_generated_images_per_step: The number of generated images per step.
            * minimum_cluster_size: The minimum cluster size.
            * target_cluster_size: The target cluster size.
            * convergence_criterion: The convergence criterion.
            * maximum_number_of_iterations: The maximum number of iterations.
        model_name: The name of the model to use.

    Returns:
        A consistent representation of the target prompt.
    """

    model_key = "CompVis/stable-diffusion-v1-4"
    # Load the specified model.
    models = {
        "stable-diffusion-v1-4": StableDiffusionPipeline.from_pretrained(model_key)
    }

    pipe = models[model_name]
    scheduler  = PNDMScheduler(model_key, subfolder="scheduler")
    for iteration in range(hyper_parameters['maximum_number_of_iterations']):
        # Generate samples from the specified model.


        samples = pipe(target_prompt, num_images=hyper_parameters['number_of_generated_images_per_step'], scheduler_type=scheduler)

        # Extract features from the samples.
        if model_name == "stable-diffusion-v1-4":
            features = samples["sample_features"]


        # Cluster the features.
        C = sp.kmeans(features, hyper_parameters['target_cluster_size'])

        # Remove small clusters.
        C = C.filter(lambda c: len(c) >= hyper_parameters['minimum_cluster_size'])

        # Find the most cohesive cluster.
        Ccohesive = min(C, key=lambda c: np.linalg.norm(c.centroid - np.mean(c, axis=0)))

        # Check the convergence criterion.
        if sp.Abs(Ccohesive.centroid[0] - Ccohesive.centroid[1]) < hyper_parameters['convergence_criterion']:
            break

    return Ccohesive

if __name__ == "__main__":
    # Set the hyper-parameters.
    hyper_parameters = {
        "number_of_generated_images_per_step": 10,
        "minimum_cluster_size": 5,
        "target_cluster_size": 10,
        "convergence_criterion": 0.01,
        "maximum_number_of_iterations": 100
    }

    # Call the consistent_character_generation() function with the specified target prompt, hyper-parameters, and model name.
    cohesive_representation = consistent_character_generation("A photo of a 50 years old man with curly hair", hyper_parameters, "stable-diffusion-v1-4")
    print(cohesive_representation)

