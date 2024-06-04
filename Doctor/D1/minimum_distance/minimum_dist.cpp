#include "minimum_dist.h"

double MinimumDist::minDist_2lines(std::vector<double>& h1, std::vector<double>& h2, std::vector<double>& r1, std::vector<double>& r2, std::vector<double>& point_h, std::vector<double>& point_r) {
	/**
	* @brief calculate minimumdistance between 2 segments of lines
	* @param[in] h1,h2 : human joints [px,py,pz], r1,r2 : robot position [px,py,pz,nx,ny,nz], nx,ny,nz : rotational vector
	* @param[out] point_h, point_r : minimum distance point
	* @return minimum distance
	*/
	std::vector<double> a0{ h1[0],h1[1],h1[2] }; std::vector<double> a1{ h2[0],h2[1],h2[2] };
	std::vector<double> b0{ r1[0],r1[1],r1[2] }; std::vector<double> b1{ r2[0],r2[1],r2[2] };
	//dirction vector
	double norm_a, norm_b;
	std::vector<double> vec_a{ a1[0] - a0[0],a1[1] - a0[1],a1[2] - a0[2] };
	std::vector<double> vec_b{ b1[0] - b0[0],b1[1] - b0[1],b1[2] - b0[2] };
	//normalize vector -> vec_a, vec_b is unit vector
	normalize(vec_a, norm_a);
	normalize(vec_b, norm_b);
	double innerDot = dot(vec_a, vec_b);
	if (abs(1-innerDot) < epsilon) //parallel
	{
		std::cout << "parallel" << std::endl;
		std::vector<double> A0B0{ b0[0] - a0[0],b0[1] - a0[1],b0[2] - a0[2] };
		std::vector<double> A0B1{ b1[0] - a0[0],b1[1] - a0[1],b1[2] - a0[2] };
		double d0 = dot(vec_a, A0B0);
		double d1 = dot(vec_a, A0B1);
		if (d0 <= 0 and d1 <= 0) {//not intersect
			if (std::abs(d0) < std::abs(d1)) { //dist between a0 and b0 is closer
				point_h = a0;
				point_r = b0;
				return distance(point_h, point_r);
			}
			else { //distance between a0 and b1 is closer
				point_h = a0;
				point_r = b1;
				return distance(point_h, point_r);
			}
		}
		else if (d0 >= norm_a and d1 >= norm_a) {//too far
			if (std::abs(d0) < std::abs(d1)) { //dist between a1 and b0 is closer
				point_h = a1;
				point_r = b0;
				return distance(point_h, point_r);
			}
			else { //distance between a0 and b1 is closer
				point_h = a1;
				point_r = b1;
				return distance(point_h, point_r);
			}
		}
		else if (0 <= d0 and d0 <= norm_a) { //choose b0
			double d = d0;
			point_h = std::vector<double>{ a0[0] + vec_a[0] * d,a0[1] + vec_a[1] * d,a0[2] + vec_a[2] * d };
			point_r = b0;
			return distance(point_h, point_r);
		}
		else if (0<=d1 and d1<norm_a){//choose b1 
			double d = d1;
			point_h = std::vector<double>{ a0[0] + vec_a[0] * d,a0[1] + vec_a[1] * d,a0[2] + vec_a[2] * d };
			point_r = b1;
			return distance(point_h, point_r);
		}
		else {
			point_h = a0;
			double dot_A0B0_vecA = dot(A0B0, vec_a);
			point_r = std::vector<double>{ a0[0] + A0B0[0]- dot_A0B0_vecA*vec_a[0],a0[1] + A0B0[1] - dot_A0B0_vecA * vec_a[1],a0[2] + A0B0[2] - dot_A0B0_vecA * vec_a[2] };
			return distance(point_h, point_r);
		}
	}
	else { //not parallel
		std::vector<double> A0B0{ b0[0] - a0[0],b0[1] - a0[1],b0[2] - a0[2] };
		double dot_ABm = dot(A0B0, vec_a);
		double dot_ABn = dot(A0B0, vec_b);
		double dot_mn = dot(vec_a, vec_b);
		//std::cout << "dot_ABm=" << dot_ABm << ", dot_ABn=" << dot_ABn << ", dot_mn=" << dot_mn << std::endl;
		double s = (dot_ABm - dot_ABn * dot_mn) / (1.0 - dot_mn * dot_mn);
		double t = (dot_ABm * dot_mn - dot_ABn) / (1.0 - dot_mn * dot_mn);
		if (std::abs(s) <= epsilon) s = 0.0;
		if (std::abs(t) <= epsilon) t = 0.0;
		//std::cout << "norm_a=" << norm_a << ", s=" << s << ", norm_b=" << norm_b << ", t=" << t << std::endl;
		point_h = std::vector<double>{ a0[0] + s * vec_a[0],a0[1] + s * vec_a[1],a0[2] + s * vec_a[2] };
		point_r = std::vector<double>{ b0[0] + t * vec_b[0],b0[1] + t * vec_b[1],b0[2] + t * vec_b[2] };
		if (method==0) {
			if ((0.0 <= s and s <= norm_a) and (0.0 <= t and t <= norm_b))//cross
				return distance(point_h, point_r);
			else//not cross
				return iterationAlgorithm(a0, b0, norm_a, norm_b, vec_a, vec_b, A0B0, point_h, point_r,0.5,0.5);
		}
		else {
			//human points
			if (s < 0.0)//minimum point isn't in segment
				point_h = a0;
			else if (s > norm_a)
				point_h = a1;
			//robot points
			if (t < 0.0)
				point_r = b0;
			else if (t > norm_b)
				point_r = b1;

			//only human is out of range -> project closest point into vector a
			double min_distance=0.0;
			if ((s<0.0 or s>norm_a) and (0.0 <= t and t <= norm_b)) {
				std::vector<double> point_temp{ point_h[0] - b0[0],point_h[1] - b0[1],point_h[2] - b0[2] };
				t = dot(vec_b, point_temp);
				if (t < 0.0)
					t = 0.0;
				else if (t > norm_b)
					t = norm_b;

				if (method==1) {
					if (s < 0.0) s = 0.0;
					else if (s > norm_a) s = norm_a;
					min_distance=iterationAlgorithm(a0, b0, norm_a, norm_b, vec_a, vec_b, A0B0, point_h, point_r, s, t);
				}
				else
					point_r = std::vector<double>{ b0[0] + t * vec_b[0],b0[1] + t * vec_b[1],b0[2] + t * vec_b[2] };
			}
			//only robot is out of range -> project closest point in vector b
			else if ((t<0.0 or t>norm_b) and (0.0 <= s and s <= norm_a)) {
				std::vector<double> point_temp{ point_r[0] - a0[0],point_r[1] - a0[1],point_r[2] - a0[2] };
				s = dot(vec_a, point_temp);
				if (s < 0.0)
					s = 0.0;
				else if (s > norm_a)
					s = norm_a;

				if (method == 1) {
					if (t < 0.0) t = 0.0;
					else if (t > norm_b) t = norm_b;
					min_distance = iterationAlgorithm(a0, b0, norm_a, norm_b, vec_a, vec_b, A0B0, point_h, point_r, s, t);
				}
				else
					point_h = std::vector<double>{ a0[0] + s * vec_a[0],a0[1] + s * vec_a[1],a0[2] + s * vec_a[2] };
			}
			else if ((s<0.0 or s>norm_a) and (t<0.0 or t>norm_b)) {
				//fix robot closest point
				std::vector<double> point_temp = std::vector<double>{ point_r[0] - point_h[0],point_r[1] - point_h[1],point_r[2] - point_h[2] };
				if (s > norm_a) vec_a = std::vector<double>{ -vec_a[0],-vec_a[1],-vec_a[2] };//in the other direction
				s = dot(vec_a, point_temp);

				if (s < 0.0)
					s = 0.0;
				else if (s > norm_a)
					s = norm_a;

				if (method == 1) {
					if (t < 0.0) t = 0.0;
					else if (t > norm_b) t = norm_b;
					min_distance = iterationAlgorithm(a0, b0, norm_a, norm_b, vec_a, vec_b, A0B0, point_h, point_r, s, t);
				}
				else {
					if (s == 0.0 or s == norm_a) {
						//fix robot closest point
						std::vector<double> point_temp{ point_h[0] - point_r[0],point_h[1] - point_r[1],point_h[2] - point_r[2] };
						if (t > norm_b) vec_b = std::vector<double>{ -vec_b[0],-vec_b[1],-vec_b[2] };//in the other direction
						t = dot(vec_b, point_temp);

						if (t < 0.0)
							t = 0.0;
						else if (t > norm_b)
							t = norm_b;
						point_r = std::vector<double>{ point_r[0] + t * vec_b[0],point_r[1] + t * vec_b[1],point_r[2] + t * vec_b[2] };
					}
					else
						point_h = std::vector<double>{ point_h[0] + s * vec_a[0],point_h[1] + s * vec_a[1],point_h[2] + s * vec_a[2] };
					
				}
					
			}
			if (method == 1)
				return min_distance;
			else {
				min_distance = distance(point_h, point_r);
				return min_distance;
			}
		}
	}
}

double MinimumDist::iterationAlgorithm(std::vector<double>& a0, std::vector<double>& b0,double& norm_a, double& norm_b, 
	std::vector<double>& vec_a, std::vector<double>& vec_b, std::vector<double>& vec_AB, std::vector<double>& point_h, 
	std::vector<double>& point_r, double s, double t) {
	/**
	* @brief calculate minimum distance and points with iteration algorithm based on gradient descent algorithm
	* @param[in] a0,b0 edge points
	* @param[in] norm_a, norm_b norm of vector_a and vector_b
	* @param[in] vec_a, vec_b unit vector for each segment of line
	* @param[in] vec_AB vector between edges of each segments
	* @param[out] point_r, point_r closest points in robot and human.
	*/

	//init parameter
	k1 = length(vec_AB);
	k1 = std::pow(k1, 2.0);//AB^2
	k2 = 2 * dot(vec_AB, vec_b);
	k3 = -2 * dot(vec_AB, vec_a);
	k4 = -2 * dot(vec_a, vec_b);
	dist_previous = 100.0;
	delta_dist = 100.0;
	delta_t = 100.0;
	delta_s = 100.0;
	//counter
	int counter = 0;
	//parameter declaration
	double f,dfdt,dfds,alpha_t,alpha_s;
	//t = find_t(norm_b);
	//execute iteration
	while (true) {
		std::cout << "counter=" << counter <<", t="<<t<<", delta_dist = " << delta_dist << ", delta_t = " << delta_t << std::endl;
		if (delta_dist < threshold_dist  or (delta_s < threshold_param and delta_t < threshold_param) or counter>=max_iteration) break;//or (delta_s < threshold_param and delta_t<threshold_param)
		
		//get optimal t and s
		//s = find_s(t, norm_a);
		//save data
		s_list.push_back(s); t_list.push_back(t);
		//calculate gradient of t
		dfdt = 2.0 * t + k4 * s + k2;
		//std::cout << "dfdt=" << dfdt <<",t="<<t <<",norm_b="<<norm_b << std::endl;
		alpha_t = get_alphaT(dfdt, t, norm_b);
		//alpha_list.push_back(alpha_t);
		t = t - alpha_t * dfdt;//update t
		delta_t = std::abs(alpha_t * dfdt);

		//calculate gradient of s
		dfds = 2.0 * s + k4 * t + k3;
		alpha_s = get_alphaS(dfds,s,norm_a);
		s = s - alpha_s * dfds;
		delta_s = std::abs(alpha_s * dfds);

		//distance
		f = function_f(s, t);
		f = std::pow(f, 0.5);
		dist_list.push_back(f);//save minimum distance
		delta_dist = dist_previous - f;
		dist_previous = f;
		counter++;
	}
	point_h = std::vector<double>{ a0[0] + s * vec_a[0],a0[1] + s * vec_a[1],a0[2] + s * vec_a[2] };
	point_r = std::vector<double>{ b0[0] + t * vec_b[0],b0[1] + t * vec_b[1],b0[2] + t * vec_b[2] };
	return f;
}

double MinimumDist::find_t(double& t_max) {
	/**
	* @brief calculate optimal t
	* @param[in] t_max maximum of t
	*/
	double num = -(k2 - k3 * k4 / 2.0);
	double den = 2.0 * (1.0 - k4 * k4 / 4.0);
	double t_candidate = num / den;
	if (t_max < t_candidate)
		return t_max;
	else if (0.0 <= t_candidate and t_candidate <= t_max)
		return t_candidate;
	else if (t_candidate < 0.0)
		return 0.0;
}
double MinimumDist::find_s(double& t, double& s_max) {
	/**
	* @brief calculate optimal t
	* @param[in] t optimized t
	* @param[out] s_max maximum of s
	*/
	double num = -(k3 + k4 * t);
	double den = 2.0;
	double s_candidate = num / den;
	if (s_max < s_candidate)
		return s_max;
	else if (0.0 <= s_candidate and s_candidate <= s_max)
		return s_candidate;
	else if (s_candidate < 0.0)
		return 0.0;

}

double MinimumDist::get_alphaS(double& dfds,double& s, double& s_max) {
	/**
	* @brief get learning rate for moving s based on gradient
	* @param[in] dfds gradient
	* @pram[in] s current s
	* @param[in] s_max maximum of s
	*/

	if (dfds <= 0.0) {//increase s
		return lambda * (s_max - s);
	}
	else if (dfds > 0.0) {//decrease s
		return lambda * s;
	}
}

double MinimumDist::get_alphaT(double& dfdt, double& t, double& t_max) {
	/**
	* @brief get learning rate for moving s based on gradient
	* @param[in] dfdt gradient
	* @pram[in] t current t
	* @param[in] t_max maximum of t
	*/

	if (dfdt <= 0.0) {//increase s
		std::cout << "move=" << (t_max - t) << std::endl;
		return lambda * (t_max - t);
	}
	else if (dfdt > 0.0) {//decrease s
		std::cout << "move=" << t << std::endl;
		return lambda * t;
	}
}

void MinimumDist::normalize(std::vector<double>& vector, double& norm) {
	/**
	* @brief normalize vector
	* @brief vector: unit vector, norm : length
	*/
	norm = std::pow(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2], 0.5); //calculate norm
	vector[0] /= norm;
	vector[1] /= norm;
	vector[2] /= norm;
}

double MinimumDist::dot(std::vector<double>& vec1, std::vector<double>& vec2) {
	/**
	* @brief inner dot
	*/
	return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

double MinimumDist::distance(std::vector<double>& point1, std::vector<double>& point2) {
	/**
	* @brief calculate distance
	*/
	double delta_x = point1[0] - point2[0];
	double delta_y = point1[1] - point2[1];
	double delta_z = point1[2] - point2[2];
	return std::pow(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z, 0.5);
}

double MinimumDist::function_f(double& s, double& t) {
	/**
	* @brief minimum distant function
	* @param[in] s,t variable
	*/
	return (k1 + s * s + t * t + k2 * t + k3 * s + k4 * s * t);
}

double MinimumDist::length(std::vector<double>& vec) {
	/**
	* @brief calculate length of vector
	* @param[in] vec vector
	*/
	double s = 0.0;
	for (double& ele : vec)
		s += ele * ele;
	s = std::pow(s, 0.5);
	return s;
}