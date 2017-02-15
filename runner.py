from src.Generate_Bands.generate_bands import gather_data, draw_fig_aritifical_models, draw_fig_cloud_sim, draw_fig_model

# To generate Band Diagrams

# pickle_file = gather_data("./Data/artificial-models/")
draw_fig_aritifical_models("./src/Generate_Bands/pickle_locker/artificial_models.p")

# pickle_file = gather_data("./Data/cloud-sim/")
# draw_fig_cloud_sim("./src/Generate_Bands/pickle_locker/cloud-sim.p")
#
# pickle_file = gather_data("./Data/models/")
# draw_fig_model("./src/Generate_Bands/pickle_locker/models.p")


# To run Rank-based method

