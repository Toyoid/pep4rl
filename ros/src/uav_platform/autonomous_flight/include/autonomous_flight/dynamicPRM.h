/*
	FILE: .h
	-----------------------------
	header of 
*/
#ifndef DYNAMIC_PRM
#define DYNAMIC_PRM
#include <autonomous_flight/flightBase.h>
#include <global_planner/dep.h>
// #include <map_manager/dynamicMap.h>
#include <map_manager/occupancyMap.h>
#include <onboard_detector/fakeDetector.h>
#include <global_planner/GetRoadmap.h>
#include <falco_planner/SetRobotPose.h>
#include <std_srvs/Empty.h>

namespace AutoFlight{
	class dynamicPRM : flightBase{
	private:
		std::shared_ptr<mapManager::occMap> map_;
		std::shared_ptr<onboardDetector::fakeDetector> detector_;
		std::shared_ptr<globalPlanner::DEP> expPlanner_;

		ros::Timer freeMapTimer_;
		ros::ServiceServer getRoadmapServer_;
		ros::ServiceServer resetRoadmapServer_;
		// ros::ServiceServer pauseMapUpdateServer_;
		
		// parameters
		bool useFakeDetector_;
		double desiredVel_;
		double desiredAcc_;
		double desiredAngularVel_;
		double wpStablizeTime_;
		bool initialScan_;

		Eigen::Vector3d freeRange_;
		double reachGoalDistance_;

		// exploration data
		bool explorationReplan_ = true;
		bool replan_ = false;
		bool newWaypoints_ = false;
		int waypointIdx_ = 1;
		nav_msgs::Path waypoints_; // latest waypoints from exploration planner

		ros::Time lastDynamicObstacleTime_;

	public:
		// std::thread exploreReplanWorker_;
		dynamicPRM();
		dynamicPRM(const ros::NodeHandle& nh);

		void initParam();
		void initModules();
	

	
		// void exploreReplan();
		void freeMapCB(const ros::TimerEvent&); // using fake detector
		bool getRoadmapServiceCB(global_planner::GetRoadmap::Request& req, global_planner::GetRoadmap::Response& resp);
		bool resetRoadmapServiceCB(falco_planner::SetRobotPose::Request& req, falco_planner::SetRobotPose::Response& resp);
		// bool pauseMapUpdateServiceCB(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp);

		void run();
		void initExplore();
	};
}
#endif