import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_score = float("inf")
    best_sum_of_marginals = None

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            temp_graph = gtsam.NonlinearFactorGraph(graph)
            temp_initial = gtsam.Values(initial_estimate)

            temp_graph, temp_initial = add_pose(temp_graph, temp_initial, pose_5)

            result = optimize(temp_graph, temp_initial)

            temp_graph = add_landmark_measurement(temp_graph, result, pose_5, landmark)

            result = optimize(temp_graph, temp_initial)

            marginals = gtsam.Marginals(temp_graph, result)

            
            score = marginals.marginalCovariance(L(landmark)).sum()

            
            total_landmark_marginals = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )

            if score < best_score:
                best_score = score
                best_pose = pose_name
                best_landmark = landmark
                best_sum_of_marginals = total_landmark_marginals

    return best_pose, best_landmark, best_sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_error_sum = float("inf")

    true_poses = {
        1: gtsam.Pose2(0.0, 0.0, 0.0),
        2: gtsam.Pose2(2.0, 0.0, 0.0),
        3: gtsam.Pose2(4.0, 0.0, 0.0),
    }

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            temp_graph = gtsam.NonlinearFactorGraph(graph)
            temp_initial = gtsam.Values(initial_estimate)

            temp_graph, temp_initial = add_pose(temp_graph, temp_initial, pose_5)

            result = optimize(temp_graph, temp_initial)

            temp_graph = add_landmark_measurement(temp_graph, result, pose_5, landmark)

            result = optimize(temp_graph, temp_initial)

            list_of_errors = []

            for pose_index in [1, 2, 3]:
                estimated_pose = result.atPose2(X(pose_index))
                true_pose = true_poses[pose_index]

                error_vector = true_pose.localCoordinates(estimated_pose)
                error = np.linalg.norm(error_vector)

                list_of_errors.append(error)

            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < best_error_sum:
                best_error_sum = sum_of_errors
                best_pose = pose_name
                best_landmark = landmark

    return best_pose, best_landmark, best_error_sum