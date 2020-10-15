/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights
   to 1.
 DO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  std::default_random_engine generator;
  std::normal_distribution<double> x_dist(x,std[0]);
  std::normal_distribution<double> y_dist(y, std[1]);
  std::normal_distribution<double> theta_dist(theta, std[2]);
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = x_dist(generator);
    particle.y = y_dist(generator);
    particle.theta = theta_dist(generator);
    particle.weight = 1;

    // append to weights, particles vector
    weights.push_back(particle.weight);
    particles.push_back(particle);
    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine generator;
  std::normal_distribution<double> x_noise_dist(0, std_pos[0]);
  std::normal_distribution<double> y_noise_dist(0, std_pos[1]);
  std::normal_distribution<double> angle_noise_dist(0, 2 * std_pos[2]); // spread the distribution of theta to get larger explore area.

  for (int i=0; i < num_particles; ++i) {
    particles[i].x += x_noise_dist(generator);
    particles[i].y += y_noise_dist(generator);
    particles[i].theta += angle_noise_dist(generator);
    double theta = particles[i].theta;
    if (abs(yaw_rate) < 1e-4) {
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta);
    } else {
      float delta_theta = yaw_rate * delta_t;
      particles[i].x += velocity / yaw_rate * (sin(theta + delta_theta) - sin(theta)) ;
      particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + delta_theta)) ;
      particles[i].theta += delta_theta;
    }

  }
}

Map::single_landmark_s ParticleFilter::dataAssociation(Map::single_landmark_s &pred_landmark,
                                                       vector<Map::single_landmark_s>& global_landmarks) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  // attention: input one predicted landmark, return the neareast ground truth landmark
  int id;
  double distance;
  Map::single_landmark_s nearest_landmark;
  double min_dist = std::numeric_limits<double>::max();  // set a large initial value for min distance
  for (auto landmark: global_landmarks) {
    distance = dist(pred_landmark.x_f, pred_landmark.y_f, landmark.x_f, landmark.y_f);
    if (distance < min_dist) {
      min_dist = distance;
      id = landmark.id_i;
      nearest_landmark = landmark;
    }
  }
  pred_landmark.id_i = id;
  return nearest_landmark;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  //double range_square = sensor_range * sensor_range;
  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];
  double sx_square = sigma_x * sigma_x;
  double sy_square = sigma_y * sigma_y;
  for (int i = 0; i < num_particles; ++i) {
    vector<Map::single_landmark_s> range_global_landmarks; // landmarks in range
    double x_part, y_part, theta, x_obs, y_obs, x_map, y_map;
    x_part = particles[i].x;
    y_part = particles[i].y;
    theta = particles[i].theta;
    // create landmark list in range
    for (auto landmark: map_landmarks.landmark_list) {
      double dist_x = landmark.x_f - x_part;
      double dist_y = landmark.y_f - y_part;
//      if ((dist_x * dist_x + dist_y * dist_y) <= range_square) {
//        range_global_landmarks.push_back(landmark);
//      }
      if ((fabs(dist_x) < sensor_range) && (fabs(dist_y) < sensor_range)) {
        range_global_landmarks.push_back(landmark);
      }
    }

    for (auto obs : observations) {
      // transform local coordinate to global coordinate
      x_obs = obs.x;
      y_obs = obs.y;
      x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
      y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);
      Map::single_landmark_s pred_global_landmark;
      pred_global_landmark.x_f = x_map;
      pred_global_landmark.y_f = y_map;
      //pred_global_landmarks.push_back(pred_global_landmark);
      Map::single_landmark_s neareast_lanmark = dataAssociation(pred_global_landmark, range_global_landmarks);

      double error_x = x_map - neareast_lanmark.x_f;
      double error_y = y_map - neareast_lanmark.y_f;
      //std::cout << " ex " << error_x << " ey " << error_y;
      double new_weight = 1/(2 * M_PI * sigma_x * sigma_y) * \
              exp(- (error_x * error_x / (2 * sx_square) + error_y * error_y / (2 * sy_square)));
      if (new_weight < 1e-4) {
        new_weight = 1e-4;
      }
      particles[i].weight *= new_weight;
    }
    weights[i] = particles[i].weight;
  }
  double w_max = 0;  // find max w
  for (auto w : weights) {
    if (w > w_max) {
      w_max = w;
    }
  }
  for (int i = 0; i < num_particles; ++i) {
    weights[i] *= exp(-log(w_max));  // normalize by divide the max weight, use multiply to avoid divide by zero
    particles[i].weight = weights[i];
    //std::cout <<" norm " << weights[i];
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  vector<Particle> resampled_particles(num_particles);
  //std:: cout << weights.size() << std::endl;
  for (int i = 0; i < num_particles; ++i) {
    //std::cout << i << std::endl;
    int id = dist(gen);
    //std::cout << "id: " << id << std::endl;
    resampled_particles[i] = particles[id];
  }
  for (int i = 0; i < num_particles; ++i) {
    particles[i] = resampled_particles[i];
  }
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}