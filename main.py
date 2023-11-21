import scipy as sp
from diffusers import StableDiffusionPipeline, PNDMScheduler
import numpy as np
import torch
from transformers import CLIPFeatureExtractor, CLIPModel
from PIL import Image
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def load_model(model_name):
    """
    Loads the specified model and returns the model pipeline.
    Args:
        model_name: The name of the model to use.
    Returns:
        The model pipeline.
    """
    models = {
        "stable-diffusion-v1-4": StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4"),
        "openai-whisper-14b": torch.hub.load('openai', 'whisper', 'https://github.com/openai/whisper/raw/main/checkpoints/whisper_14b.pt')
    }
    return models.get(model_name, None)


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

    # scheduler  =  PNDMScheduler(
    #         beta_start=0.00085,
    #         beta_end=0.012,
    #         beta_schedule="scaled_linear",
    #         skip_prk_steps = True)
    previous_centroid = None
    most_cohesive_cluster_info = None

    for iteration in range(hyper_parameters['maximum_number_of_iterations']):
        result = pipe(target_prompt, num_images=hyper_parameters['number_of_generated_images_per_step'])
        inputs = feature_extractor(images=result.images, return_tensors="pt")
        features = model.get_image_features(**inputs)
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features

        # Use KMeans for clustering
        kmeans = KMeans(n_clusters=hyper_parameters['target_cluster_size'])
        kmeans.fit(features_np)

        # Calculate the average distance of points to their cluster center
        cluster_distances = euclidean_distances(features_np, kmeans.cluster_centers_)
        avg_distances = np.array([cluster_distances[kmeans.labels_ == i].mean() for i in range(kmeans.n_clusters)])

        # Find the index of the most cohesive cluster
        most_cohesive_index = np.argmin(avg_distances)
        most_cohesive_centroid = kmeans.cluster_centers_[most_cohesive_index]

        # Check the convergence criterion
        if previous_centroid is not None:
            centroid_diff = np.linalg.norm(most_cohesive_centroid - previous_centroid)
            if centroid_diff < hyper_parameters['convergence_criterion']:
                break

        previous_centroid = most_cohesive_centroid
        # Update the most cohesive cluster info
        most_cohesive_cluster_info = {
            'centroid': most_cohesive_centroid,
            'labels': kmeans.labels_,
            'cohesive_cluster_label': most_cohesive_index
        }
    
    return most_cohesive_cluster_info 

def extract_features(image_path, model):
    # Load the image. This will depend on how your model expects the input.
    image = load_image(image_path)

    # Process the image through the model to get features.
    # This is a simplified example; the actual method depends on your model's API.
    features = model(image)

    return features

def load_image(image_path):
    # Load the image from the path.
    # Implement this according to your model's input requirements.
    pass



def compare_features(image_features, cluster_centroid):
    # Calculate the Euclidean distance between the two feature vectors
    distance = np.linalg.norm(image_features - cluster_centroid)

    return distance


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

    def evaluate_cohesiveness(new_image_path, model, cohesive_cluster):
        # Load and process the new image to extract features
        new_image_features = extract_features(new_image_path, model)

        # Compare the new image's features to the cohesive cluster's centroid
        cohesiveness_score = compare_features(new_image_features, cohesive_cluster.centroid)

        return cohesiveness_score

    # Usage
    pipe = load_model("stable-diffusion-v1-4")
    new_image_cohesiveness = evaluate_cohesiveness("path_to_new_image.jpg", pipe, cohesive_representation)
    print(f"Cohesiveness score: {new_image_cohesiveness}")
