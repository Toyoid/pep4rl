/*
	FILE: .cpp
	-----------------------------
	ROS node for testing incremental construction of probabilistic roadmap
*/

#include <autonomous_flight/dprmPlanManager.h>

int main(int argc, char** argv){
	ros::init(argc, argv, "dprm_planner_node");
	ros::NodeHandle nh;
	AutoFlight::dprmPlanManager dprm_planner (nh);
	dprm_planner.run();

	ros::spin();

	return 0;
}