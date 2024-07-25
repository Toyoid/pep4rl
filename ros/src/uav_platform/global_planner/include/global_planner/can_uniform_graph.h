/*
*	File: dep.h
*	---------------
*   dynamic exploration planner header file
*/

#ifndef CAN_UNIFORM_GRAPH_H
#define CAN_UNIFORM_GRAPH_H

#include <map_manager/dynamicMap.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>  //added by me
#include <gazebo_msgs/ModelState.h>  //added
#include <global_planner/PRMKDTree.h>
#include <global_planner/PRMAstar.h>
#include <global_planner/utils.h>
#include <global_planner/point_struct.h>
#include <opencv2/opencv.hpp>


namespace globalPlanner{
    // self-defined hash function for Eigen::Vector3d
    struct Vector3dHash {
        std::size_t operator()(const Eigen::Vector3d& vec) const {
            std::size_t hx = std::hash<double>()(vec.x());
            std::size_t hy = std::hash<double>()(vec.y());
            std::size_t hz = std::hash<double>()(vec.z());
            return hx ^ (hy << 1) ^ (hz << 2);
        }
    };

    // self-defined equal-to function for Eigen::Vector3d
    struct Vector3dEqual {
        bool operator()(const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs) const {
            return lhs.isApprox(rhs);
        }
    };
    
	class CAN{
	private:
		std::string ns_;
		std::string hint_;

		ros::NodeHandle nh_;
		ros::Publisher roadmapPub_;
		ros::Publisher bestPathPub_;
		ros::Publisher waypointPub_;  //added by me
		ros::Subscriber odomSub_;
		ros::Subscriber currGoalSub_;  //added by me
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
		double safeDistXY_;
		double safeDistZ_;
		bool safeDistCheckUnknown_;
		double horizontalFOV_;
		double verticalFOV_;
		double dmin_;
		double dmax_;
		int nnNum_;
		std::vector<double> yaws_;
		double minVoxelThresh_;

        double graphNodeHeight_;
        std::vector<int> numGlobalPoints_;
        Eigen::Vector3d envGlobalRangeMax_, envGlobalRangeMin_;


		// data
		bool odomReceived_ = false;
		bool currGoalReceived_ = false;  //added by me
		bool resettingRLEnv_ = false; //added
		Eigen::Vector3d position_;
		std::shared_ptr<PRM::Node> currGoal_;  //added by me
		std::shared_ptr<PRM::Node> ultimateTarget_;
		unsigned int waypointIdx_ = 0;  //added by me
		Point3D navWaypoint_;  //added by me
		Point3D navHeading_;  //added by me
		double currYaw_;
		std::deque<Eigen::Vector3d> histTraj_; // historic trajectory for information gain update 	
		std::unordered_set<std::shared_ptr<PRM::Node>> canNodeVec_; // all nodes
		std::unordered_set<Eigen::Vector3d, Vector3dHash, Vector3dEqual> uncoveredGlobalPoints_; // uncovered uniform nodes
		std::vector<std::shared_ptr<PRM::Node>> bestPath_;


	public:
		CAN(const ros::NodeHandle& nh);

		void setMap(const std::shared_ptr<mapManager::occMap>& map);
		void loadVelocity(double vel, double angularVel);
		void initParam();
		void initModules();
		void registerPub();
		void registerCallback();

		bool makePlan();
		void buildRoadMap();
		void pruneNodes();
		void updateInformationGain();		

		// callback functions
		void odomCB(const nav_msgs::OdometryConstPtr& odom);
		void visCB(const ros::TimerEvent&);
		void currGoalCB(const geometry_msgs::PointStamped::ConstPtr& goal);  
 		void waypointUpdateCB(const ros::TimerEvent&);  
		void ultimateTargetCB(const geometry_msgs::PointStamped::ConstPtr& navTarget);

		// help function
		bool isPosValid(const Eigen::Vector3d& p);
		bool isPosValid(const Eigen::Vector3d& p, double safeDistXY, double safeDistZ);
        bool isNodeRequireUpdate(std::shared_ptr<PRM::Node> n, std::vector<std::shared_ptr<PRM::Node>> path, double& leastDistance);
		bool sensorFOVCondition(const Eigen::Vector3d& sample, const Eigen::Vector3d& pos);
		int calculateUnknown(const shared_ptr<PRM::Node>& n, std::unordered_map<double, int>& yawNumVoxels);
		double calculatePathLength(const std::vector<shared_ptr<PRM::Node>>& path);
		void shortcutPath(const std::vector<std::shared_ptr<PRM::Node>>& path, std::vector<std::shared_ptr<PRM::Node>>& pathSc);
		Point3D projectNavWaypoint(const Point3D& nav_waypoint, const Point3D& last_waypoint);  // added by me

		// visualization functions
		visualization_msgs::MarkerArray buildRoadmapMarkers();
		void publishBestPath();

		// clear function for RL training
		void resetRoadmap(const gazebo_msgs::ModelState& resetRobotPos);

        std::unordered_set<Eigen::Vector3d, Vector3dHash, Vector3dEqual> generateGlobalPoints2D(double& height);
	};
}


#endif


