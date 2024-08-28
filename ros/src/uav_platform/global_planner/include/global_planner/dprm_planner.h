/*
*	File: dprm_planner.h
*	---------------
*   dynamic PRM global planner header file
*/

#ifndef DPRM_PLANNER_H
#define DPRM_PLANNER_H

#include <map_manager/dynamicMap.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/PointStamped.h>  //added by me
#include <gazebo_msgs/ModelState.h>  //added
#include <global_planner/PRMKDTree.h>
#include <global_planner/PRMAstar.h>
#include <global_planner/utils.h>
#include <global_planner/time_measure.h>
#include <global_planner/point_struct.h>
#include <opencv2/opencv.hpp>


namespace globalPlanner{
	class DPRM{
	private:
		std::string ns_;
		std::string hint_;

		ros::NodeHandle nh_;
		ros::Publisher roadmapPub_;
		ros::Publisher bestPathPub_;
		ros::Publisher frontierVisPub_;
		ros::Publisher waypointPub_;  //added by me
		ros::Publisher currGoalPub_;
		ros::Publisher runtimePub_;
		ros::Subscriber odomSub_;
		ros::Subscriber ultimateTargetSub_;
		ros::Timer visTimer_;
		ros::Timer waypointTimer_;  //added by me

		nav_msgs::Odometry odom_;
		std::shared_ptr<mapManager::occMap> map_; 
		std::shared_ptr<PRM::KDTree> roadmap_;

		// parameters
		double vel_ = 1.0;
		double angularVel_ = 1.0;
		std::string odomTopic_;
		Eigen::Vector3d globalRegionMin_, globalRegionMax_;
		Eigen::Vector3d localRegionMin_, localRegionMax_;
		int localSampleThresh_;
		int globalSampleThresh_;
		int frontierSampleThresh_;
		double distThresh_;
		double safeDistXY_;
		double safeDistZ_;
		bool safeDistCheckUnknown_;
		double horizontalFOV_;
		double verticalFOV_;
		double dmin_;
		double dmax_;
		int nnNum_;
		int nnNumFrontier_;
		double maxConnectDist_;
		std::vector<double> yaws_;
		double minVoxelThresh_;
		int minCandidateNum_;
		int maxCandidateNum_;
		double updateDist_;
		double yawPenaltyWeight_;
		double dist2TargetWeight_;
		double pathLengthWeight_;

		// data
		bool odomReceived_ = false;
		bool ultimateTargetReceived_ = false;  
		bool currGoalRecieved_ = false;  
		bool resettingRLEnv_ = false; 
		Eigen::Vector3d position_;
		Eigen::Vector3d stopPos_;
		std::shared_ptr<PRM::Node> currGoal_;
		std::shared_ptr<PRM::Node> ultimateTarget_;
		unsigned int waypointIdx_ = 0;  //added by me
		Point3D navWaypoint_;  //added by me
		Point3D navHeading_;  //added by me
		double currYaw_;
		std::deque<Eigen::Vector3d> histTraj_; // historic trajectory for information gain update 
		// std::vector<std::shared_ptr<PRM::Node>> prmNodeVec_; // all nodes		
		std::unordered_set<std::shared_ptr<PRM::Node>> prmNodeVec_; // all nodes
		std::vector<std::shared_ptr<PRM::Node>> goalCandidates_;
		std::vector<std::vector<std::shared_ptr<PRM::Node>>> candidatePaths_;
		std::vector<std::shared_ptr<PRM::Node>> bestPath_;
		std::vector<std::pair<Eigen::Vector3d, double>> frontierPointPairs_;
		// time measure
		TimeMeasure measureTimer_;
		std_msgs::Float32 runtime_;

	public:
		DPRM(const ros::NodeHandle& nh);

		void setMap(const std::shared_ptr<mapManager::occMap>& map);
		void loadVelocity(double vel, double angularVel);
		void initParam();
		void initModules();
		void registerPub();
		void registerCallback();
		bool isUltimateTargetReceived();
		void setCurrGoalRecieved(bool& currGoalReceived);

		bool makePlan();
		nav_msgs::Path getBestPath();
		void detectFrontierRegion(std::vector<std::pair<Eigen::Vector3d, double>>& frontierPointPairs);
		void buildRoadMap();
		void pruneNodes();
		void getBestCurrGoal();
		void updateInformationGain();
		void getBestViewCandidates(std::vector<std::shared_ptr<PRM::Node>>& goalCandidates);
		bool findCandidatePath(const std::vector<std::shared_ptr<PRM::Node>>& goalCandidates,  std::vector<std::vector<std::shared_ptr<PRM::Node>>>& candidatePaths);
		void findBestPath(const std::vector<std::vector<std::shared_ptr<PRM::Node>>>& candidatePaths, std::vector<std::shared_ptr<PRM::Node>>& bestPath);		

		// callback functions
		void odomCB(const nav_msgs::OdometryConstPtr& odom);
		void visCB(const ros::TimerEvent&);
 		void waypointUpdateCB(const ros::TimerEvent&);  
		void ultimateTargetCB(const geometry_msgs::PointStamped::ConstPtr& navTarget);

		// help function
		bool isPosValid(const Eigen::Vector3d& p);
		bool isPosValid(const Eigen::Vector3d& p, double safeDistXY, double safeDistZ);
		bool isNodeRequireUpdate(std::shared_ptr<PRM::Node> n, std::vector<std::shared_ptr<PRM::Node>> path, double& leastDistance);
		std::shared_ptr<PRM::Node> randomConfigBBox(const Eigen::Vector3d& minRegion, const Eigen::Vector3d& maxRegion);
		bool sensorRangeCondition(const shared_ptr<PRM::Node>& n1, const shared_ptr<PRM::Node>& n2);
		bool sensorFOVCondition(const Eigen::Vector3d& sample, const Eigen::Vector3d& pos);
		int calculateUnknown(const shared_ptr<PRM::Node>& n, std::unordered_map<double, int>& yawNumVoxels);
		double calculatePathLength(const std::vector<shared_ptr<PRM::Node>>& path);
		void shortcutPath(const std::vector<std::shared_ptr<PRM::Node>>& path, std::vector<std::shared_ptr<PRM::Node>>& pathSc);
		int weightedSample(const std::vector<double>& weights);
		std::shared_ptr<PRM::Node> sampleFrontierPoint(const std::vector<double>& sampleWeights);
		std::shared_ptr<PRM::Node> extendNode(const std::shared_ptr<PRM::Node>& n, const std::shared_ptr<PRM::Node>& target);
		Point3D projectNavWaypoint(const Point3D& nav_waypoint, const Point3D& last_waypoint);  // added by me

		// visualization functions
		visualization_msgs::MarkerArray buildRoadmapMarkers();
		void publishCandidatePaths();
		void publishBestPath();
		void publishFrontier();

		// clear function for RL training
		void resetRoadmap(const gazebo_msgs::ModelState& resetRobotPos);
	};
}


#endif


