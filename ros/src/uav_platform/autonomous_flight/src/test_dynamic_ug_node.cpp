/*
	FILE: .cpp
	-----------------------------
	ROS node for testing incremental construction of probabilistic roadmap
*/

#include <autonomous_flight/dynamicUG.h>  // dynamic uniform graph

int main(int argc, char** argv){
	ros::init(argc, argv, "dynamic_ug_node");
	ros::NodeHandle nh;
	AutoFlight::dynamicUG d_ug (nh);
	d_ug.run();

	ros::spin();

	return 0;
}