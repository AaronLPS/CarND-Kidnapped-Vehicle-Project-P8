/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Implement the particle filter on August 2018
 *      Author: Aaron Shi
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;
    //Define gaussian noise
    normal_distribution<double> noise_x(0, std[0]);
    normal_distribution<double> noise_y(0, std[1]);
    normal_distribution<double> noise_theta(0, std[2]);

    //Initalize particles
    for(int i=0; i<num_particles; i++){
        Particle new_p;
        new_p.id = i;
        new_p.x = x;
        new_p.y = y;
        new_p.theta = theta;
        new_p.weight = 1.0;
        //Add radom Gaussian noise
        new_p.x += noise_x(gen);
        new_p.y += noise_y(gen);
        new_p.theta += noise_theta(gen);

        particles.push_back(new_p);

        is_initialized = true;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    //Define gaussian noise
    normal_distribution<double> noise_x(0, std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);

    //Predict new status based on motion model CTRV
    for(int i=0; i<num_particles; i++){
        //deal with yaw rate = 0
        if (fabs(yaw_rate) < 0.00001) {
          particles[i].x += velocity * delta_t * cos(particles[i].theta);
          particles[i].y += velocity * delta_t * sin(particles[i].theta);
          //theta keeps the same
        }
        else {
          particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
          particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
          particles[i].theta += yaw_rate * delta_t;
        }

        //Add random Gaussion Noise
        particles[i].x += noise_x(gen);
        particles[i].y += noise_y(gen);
        particles[i].theta += noise_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    // Nearest Neighbour method complexity: O(ij)
    for(int i=0; i<observations.size(); i++){
        LandmarkObs current_landmarkOb = observations[i];

        //initialize with the first predict
        LandmarkObs current_predict = predicted[0];
        double distance = dist(current_landmarkOb.x, current_landmarkOb.y, current_predict.x, current_predict.y);
        double min_distance = distance;
        int associated_id = current_predict.id;

        //process the following predicts
        for (int j=1; j<predicted.size(); j++){
            current_predict = predicted[j];

            //calculate the distance
            distance = dist(current_landmarkOb.x, current_landmarkOb.y, current_predict.x, current_predict.y);
            //update the min_distance
            if(distance < min_distance){
                min_distance = distance;
                associated_id = current_predict.id;
            }
        }
        //link the id to the observation
        observations[i].id = associated_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // for each particle:
    // The particles final weight will be calculated as the product of each
    // measurement's Multivariate-Gaussian probability densit
    for(int i=0; i < num_particles; i++){
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        // Build a vector to store the landmarks,
        // which are predicted to be within sensor_range around the target particle (roi)
        vector<LandmarkObs> landmark_roi;
        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            // get id and x,y coordinates
            float lm_x = map_landmarks.landmark_list[j].x_f;
            float lm_y = map_landmarks.landmark_list[j].y_f;
            int lm_id = map_landmarks.landmark_list[j].id_i;
            // consider landmarks within sensor range of the particle
            // the range is defined as a rectangle, (can also be a circle here)
            if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {
              // add landmarks inside ROI to vector
              landmark_roi.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
            }
        }

        // Transform observation form vehicle coordinates to map coordinates.
        // Creat a vector to store the transformed observations
        vector<LandmarkObs> transformed_obs;
        for (unsigned int j = 0; j < observations.size(); j++) {
            double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
            double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
            transformed_obs.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
        }

        // Data association, link landmark with the nearest observation
        dataAssociation(landmark_roi, transformed_obs);

        // reset the weight of particle before the start of each round of weight update
        particles[i].weight = 1.0;

        // calculate measurement's Multivariate-Gaussian probability densit
        // then, product all of them
        for (int j = 0; j < transformed_obs.size(); j++) {

            // placeholders for observation and associated landmark coordinates
            double o_x, o_y, lm_x, lm_y;
            o_x = transformed_obs[j].x;
            o_y = transformed_obs[j].y;

            // find the x,y coordinates of the landmark associated with the observation
            for (int k = 0; k < landmark_roi.size(); k++) {
                if (landmark_roi[k].id == transformed_obs[j].id) {
                  lm_x = landmark_roi[k].x;
                  lm_y = landmark_roi[k].y;
                }
            }

            // Multivariate Gaussian
            double s_x = std_landmark[0];
            double s_y = std_landmark[1];
            double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(lm_x-o_x,2)/(2*pow(s_x, 2)) + (pow(lm_y-o_y,2)/(2*pow(s_y, 2))) ) );

            // product all
            particles[i].weight *= obs_w;
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    //creat vector to store resampled particles
    vector<Particle> new_particles;

    // list all weights
    vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    // generate random index as start
    uniform_int_distribution<int> random_gen_index(0, num_particles-1);
    int index = random_gen_index(gen);

    // get the max weight
    double max_weight = *max_element(weights.begin(), weights.end());

    // random distribution [0.0, max_weight)
    uniform_real_distribution<double> random_gen_distr(0.0, max_weight);

    double beta = 0.0;
    // resampling wheel spin for num_particles times
    for (int i = 0; i < num_particles; i++) {
        beta += random_gen_distr(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


