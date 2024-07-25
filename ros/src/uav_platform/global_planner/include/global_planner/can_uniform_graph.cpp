/*
*	File: dep.cpp
*	---------------
*   dynamic exploration planner implementation
*/

#include <global_planner/can_uniform_graph.h>
#include <random>


namespace globalPlanner{
	CAN::CAN(const ros::NodeHandle& nh) : nh_(nh){
		this->ns_ = "/CAN";
		this->hint_ = "[CAN]";
		this->initParam();
		this->initModules();
		this->registerPub();
		this->registerCallback();
	}

	void CAN::setMap(const std::shared_ptr<mapManager::occMap>& map){
		this->map_ = map;
	}

	void CAN::loadVelocity(double vel, double angularVel){
		this->vel_ = vel;
		this->angularVel_ = angularVel;
	}

	void CAN::initParam(){
	    // initialize ultimate navigation target
		this->ultimateTarget_.reset(new PRM::Node (Eigen::Vector3d(0, 0, -3)));

		// odom topic name
		if (not this->nh_.getParam(this->ns_ + "/odom_topic", this->odomTopic_)){
			this->odomTopic_ = "/CERLAB/quadcopter/odom";
			cout << this->hint_ << ": No odom topic name. Use default: /CERLAB/quadcopter/odom" << endl;
		}
		else{
			cout << this->hint_ << ": Odom topic: " << this->odomTopic_ << endl;
		}

		// global sample region min
		std::vector<double> globalRegionMinTemp;	
		if (not this->nh_.getParam(this->ns_ + "/global_region_min", globalRegionMinTemp)){
			this->globalRegionMin_(0) = -20.0;
			this->globalRegionMin_(1) = -20.0;
			this->globalRegionMin_(2) = 0.0;
			cout << this->hint_ << ": No global region min param. Use default: [-20 -20 0]" <<endl;
		}
		else{
			this->globalRegionMin_(0) = globalRegionMinTemp[0];
			this->globalRegionMin_(1) = globalRegionMinTemp[1];
			this->globalRegionMin_(2) = globalRegionMinTemp[2];
			cout << this->hint_ << ": Global Region Min: " << this->globalRegionMin_[0] <<" "<< this->globalRegionMin_[1]<<" "<< this->globalRegionMin_[2]<< endl;
		}

		// global sample region max
		std::vector<double> globalRegionMaxTemp;	
		if (not this->nh_.getParam(this->ns_ + "/global_region_max", globalRegionMaxTemp)){
			this->globalRegionMax_(0) = 20.0;
			this->globalRegionMax_(1) = 20.0;
			this->globalRegionMax_(2) = 3.0;
			cout << this->hint_ << ": No global region max param. Use default: [20 20 3]" <<endl;
		}
		else{
			this->globalRegionMax_(0) = globalRegionMaxTemp[0];
			this->globalRegionMax_(1) = globalRegionMaxTemp[1];
			this->globalRegionMax_(2) = globalRegionMaxTemp[2];
			cout << this->hint_ << ": Global Region Max: " << this->globalRegionMax_[0] <<" "<< this->globalRegionMax_[1]<<" "<< this->globalRegionMax_[2]<< endl;
		}	

		// safety distance for random sampling in xy
		if (not this->nh_.getParam(this->ns_ + "/safe_distance_xy", this->safeDistXY_)){
			this->safeDistXY_ = 0.3;
			cout << this->hint_ << ": No safe distance in XY param. Use default: 0.3" << endl;
		}
		else{
			cout << this->hint_ << ": Safe distance in XY: " << this->safeDistXY_ << endl;
		}

		// safety distance for random sampling in z
		if (not this->nh_.getParam(this->ns_ + "/safe_distance_z", this->safeDistZ_)){
			this->safeDistZ_ = 0.0;
			cout << this->hint_ << ": No safe distance in Z param. Use default: 0.0" << endl;
		}
		else{
			cout << this->hint_ << ": Safe distance in Z: " << this->safeDistZ_ << endl;
		}

		// safety distance check unknown
		if (not this->nh_.getParam(this->ns_ + "/safe_distance_check_unknown", this->safeDistCheckUnknown_)){
			this->safeDistCheckUnknown_ = true;
			cout << this->hint_ << ": No safe distance check unknown param. Use default: true" << endl;
		}
		else{
			cout << this->hint_ << ": Safe distance check unknown: " << this->safeDistCheckUnknown_ << endl;
		}

		//Camera Parameters	
		if (not this->nh_.getParam(this->ns_ + "/horizontal_FOV", this->horizontalFOV_)){
			this->horizontalFOV_ = 0.8;
			cout << this->hint_ << ": No Horizontal FOV param. Use default: 0.8" << endl;
		}
		else{
			cout << this->hint_ << ": Horizontal FOV: " << this->horizontalFOV_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/vertical_FOV", this->verticalFOV_)){
			this->verticalFOV_ = 0.8;
			cout << this->hint_ << ": No Vertical FOV param. Use default: 0.8" << endl;
		}
		else{
			cout << this->hint_ << ": Vertical FOV: " << this->verticalFOV_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/dmin", this->dmin_)){
			this->dmin_ = 0.0;
			cout << this->hint_ << ": No min depth param. Use default: 0.0" << endl;
		}
		else{
			cout << this->hint_ << ": Min Depth: " << this->dmin_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/dmax", this->dmax_)){
			this->dmax_ = 1.0;
			cout << this->hint_ << ": No max depth param. Use default: 1.0" << endl;
		}
		else{
			cout << this->hint_ << ": Max Depth: " << this->dmax_ << endl;
		}

		// nearest neighbor number
		if (not this->nh_.getParam(this->ns_ + "/nearest_neighbor_number", this->nnNum_)){
			this->nnNum_ = 20;
			cout << this->hint_ << ": No nearest neighbor param. Use default: 20" << endl;
		}
		else{
			cout << this->hint_ << ": Nearest neighbor number is set to: " << this->nnNum_ << endl;
		}

		// number of yaw angles
		int yawNum = 32;
		if (not this->nh_.getParam(this->ns_ + "/num_yaw_angles", yawNum)){
			for (int i=0; i<32; ++i){
				this->yaws_.push_back(i*2*PI_const/32);
			}					
			cout << this->hint_ << ": No number of yaw angles param. Use default: 32." << endl;
		}
		else{
			for (int i=0; i<yawNum; ++i){
				this->yaws_.push_back(i*2*PI_const/32);
			}	
			cout << this->hint_ << ": Number of yaw angles is set to: " << yawNum << endl;
		}

        if (not this->nh_.getParam(this->ns_ + "/graph_node_height", this->graphNodeHeight_)){
			this->graphNodeHeight_ = 1.5;
			cout << this->hint_ << ": No graph node height param. Use default: 1.5." << endl;
		}
		else{
			cout << this->hint_ << ": Graph node height is set to: " << this->graphNodeHeight_ << endl;
		}

        std::vector<int> numGlobalPointsTemp;	
		if (not this->nh_.getParam(this->ns_ + "/num_global_points", numGlobalPointsTemp)){
			this->numGlobalPoints_.push_back(30);
			this->numGlobalPoints_.push_back(30);
			this->numGlobalPoints_.push_back(3);
			cout << this->hint_ << ": No num global points param. Use default: [20 20 3]" <<endl;
		}
		else{
			this->numGlobalPoints_.push_back(numGlobalPointsTemp[0]);
			this->numGlobalPoints_.push_back(numGlobalPointsTemp[1]);
			this->numGlobalPoints_.push_back(numGlobalPointsTemp[2]);
			cout << this->hint_ << ": Number of global points: " << this->numGlobalPoints_[0] <<" "<< this->numGlobalPoints_[1]<<" "<< this->numGlobalPoints_[2]<< endl;
		}

        std::vector<double> envGlobalRangeMaxTemp;
        if (not this->nh_.getParam(this->ns_ + "/env_global_range_max", envGlobalRangeMaxTemp)) {
            this->envGlobalRangeMax_(0) = 10.0;
            this->envGlobalRangeMax_(1) = 10.0;
            this->envGlobalRangeMax_(2) = 2.0;
            cout << this->hint_ << ": No env global range max param. Use default: [10.0 10.0 2.0]" << endl;
        } else {
            this->envGlobalRangeMax_(0) = envGlobalRangeMaxTemp[0];
            this->envGlobalRangeMax_(1) = envGlobalRangeMaxTemp[1];
            this->envGlobalRangeMax_(2) = envGlobalRangeMaxTemp[2];
            cout << this->hint_ << ": Global range max: " << this->envGlobalRangeMax_[0] << " " << this->envGlobalRangeMax_[1] << " " << this->envGlobalRangeMax_[2] << endl;
        }

        std::vector<double> envGlobalRangeMinTemp;
        if (not this->nh_.getParam(this->ns_ + "/env_global_range_min", envGlobalRangeMinTemp)) {
            this->envGlobalRangeMin_(0) = -10.0;
            this->envGlobalRangeMin_(1) = -10.0;
            this->envGlobalRangeMin_(2) = 0.5;
            cout << this->hint_ << ": No env global range min param. Use default: [-10.0 -10.0 0.5]" << endl;
        } else {
            this->envGlobalRangeMin_(0) = envGlobalRangeMinTemp[0];
            this->envGlobalRangeMin_(1) = envGlobalRangeMinTemp[1];
            this->envGlobalRangeMin_(2) = envGlobalRangeMinTemp[2];
            cout << this->hint_ << ": Global range min: " << this->envGlobalRangeMin_[0] << " " << this->envGlobalRangeMin_[1] << " " << this->envGlobalRangeMin_[2] << endl;
        }


        // initialize global uniform graph
        this->uncoveredGlobalPoints_ = this->generateGlobalPoints2D(this->graphNodeHeight_);

	}

	void CAN::initModules(){
		// initialize roadmap
		this->roadmap_.reset(new PRM::KDTree ());
	}

	void CAN::resetRoadmap(const gazebo_msgs::ModelState& resetRobotPos) {
	    // IMPORTANT: lock waypointUpdateTimer to prevent visiting invalid memory
	    this->currGoalReceived_ = false;
		this->resettingRLEnv_ = false;

		this->canNodeVec_.clear();
		this->roadmap_->clear();
        this->uncoveredGlobalPoints_ = this->generateGlobalPoints2D(this->graphNodeHeight_);

		Eigen::Vector3d p;
		p(0) = resetRobotPos.pose.position.x;
		p(1) = resetRobotPos.pose.position.y;
		p(2) = resetRobotPos.pose.position.z;
		this->currGoal_.reset(new PRM::Node(p));

		this->navWaypoint_ = Point3D(p(0), p(1), p(2));
		this->navHeading_ = Point3D(0, 0, 0);

		this->waypointIdx_ = 0;  
		this->histTraj_.clear();  
		this->bestPath_.clear();

	    // IMPORTANT: unlock waypointUpdateTimer
		this->currGoalReceived_ = false;
		this->resettingRLEnv_ = true;
	}

	void CAN::registerPub(){
		// roadmap visualization publisher
		this->roadmapPub_ = this->nh_.advertise<visualization_msgs::MarkerArray>("/dep/roadmap", 5);

		// best path publisher
		this->bestPathPub_ = this->nh_.advertise<visualization_msgs::MarkerArray>("/dep/best_paths", 10);

		// waypoint publisher
		this->waypointPub_ = this->nh_.advertise<geometry_msgs::PointStamped>("/falco_planner/way_point", 5); 
	}

	void CAN::registerCallback(){
		// odom subscriber
		this->odomSub_ = this->nh_.subscribe(this->odomTopic_, 1000, &CAN::odomCB, this);
		// this->odomSub_ = this->nh_.subscribe("/falco_planner/state_estimation", 1000, &CAN::odomCB, this);

		// ultimate target subscriber
		this->ultimateTargetSub_ = this->nh_.subscribe("/env/nav_target", 5, &CAN::ultimateTargetCB, this);
	
		// visualization timer
		this->visTimer_ = this->nh_.createTimer(ros::Duration(0.2), &CAN::visCB, this);

		//added by me waypoint publish timer
		this->waypointTimer_ = this->nh_.createTimer(ros::Duration(0.33), &CAN::waypointUpdateCB, this);

		//added by me
		this->currGoalSub_ = this->nh_.subscribe("/agent/current_goal", 5, &CAN::currGoalCB, this); 
	}

    std::unordered_set<Eigen::Vector3d, Vector3dHash, Vector3dEqual> CAN::generateGlobalPoints2D(double& height) {
        std::vector<double> x(this->numGlobalPoints_[0]), y(this->numGlobalPoints_[1]);

        for (int i = 1; i <= this->numGlobalPoints_[0]; ++i) {
            x[i-1] = (this->envGlobalRangeMax_(0) - this->envGlobalRangeMin_(0)) * i / this->numGlobalPoints_[0] + this->envGlobalRangeMin_(0);
        }
        for (int i = 1; i <= this->numGlobalPoints_[1]; ++i) {
            y[i-1] = (this->envGlobalRangeMax_(1) - this->envGlobalRangeMin_(1)) * i / this->numGlobalPoints_[1] + this->envGlobalRangeMin_(1);
        }

        std::unordered_set<Eigen::Vector3d, Vector3dHash, Vector3dEqual> points;

        for (const auto& xi : x) {
            for (const auto& yi : y) {
                points.emplace(Eigen::Vector3d(xi, yi, height));
            }
        }

        return points;
    }

	bool CAN::makePlan(){
		if (not this->odomReceived_) return false;
	
		this->buildRoadMap();
	
		this->pruneNodes();

		this->updateInformationGain();

		return true;
	}

	// create sensor check for 
	// vert, horz FOV, collision, and sensor distance range
	// for yaw angles in vector3d:  cos(yaw), sin(yaw), 0
	// horz angle between yaw angle vector and direction (x y 0) vector for horz FOV
	// Vert angle yaw angle vector and yaw angle vector (c s z) z is direction.z()
	bool CAN::sensorFOVCondition(const Eigen::Vector3d& sample, const Eigen::Vector3d& pos){
		Eigen::Vector3d direction = sample - pos;
		double distance = direction.norm();
		if (distance > this->dmax_){
			return false;
		}
		bool hasCollision = this->map_->isInflatedOccupiedLine(sample, pos);
		if (hasCollision == true){
			return false;
		}
		return true;
	}

	bool CAN::isNodeRequireUpdate(std::shared_ptr<PRM::Node> n, std::vector<std::shared_ptr<PRM::Node>> path, double& leastDistance){
		double distanceThresh = 2;
		leastDistance = std::numeric_limits<double>::max();
		for (std::shared_ptr<PRM::Node>& waypoint: path){
			double currentDistance = (n->pos - waypoint->pos).norm();
			if (currentDistance < leastDistance){
				leastDistance = currentDistance;
			}
		}
		if (leastDistance <= distanceThresh){
			return true;
		}
		else{
			return false;	
		}
		
	}

	void CAN::buildRoadMap(){
		std::shared_ptr<PRM::Node> n;
		std::vector<std::shared_ptr<PRM::Node>> newNodes;

		// add ultimate navigation goal to PRM
		if (this->isPosValid(this->ultimateTarget_->pos, this->safeDistXY_, this->safeDistZ_) && (!this->canNodeVec_.count(this->ultimateTarget_))) {
			this->canNodeVec_.insert(this->ultimateTarget_);
			this->roadmap_->insert(this->ultimateTarget_);
			newNodes.push_back(this->ultimateTarget_);
		}

        // 1. scan the uncoverd global points
        // 2. insert the ones that within the explored map, and erase them from the uncovered global points 
        for (const Eigen::Vector3d& point: this->uncoveredGlobalPoints_) {
            Eigen::Vector3d diff = point - this->ultimateTarget_->pos;
            double dist = diff.norm();
            bool addToGraph = (dist >= 0.2);
            if (addToGraph && this->isPosValid(point, this->safeDistXY_, this->safeDistZ_)) {
                std::shared_ptr<PRM::Node> newNode (new PRM::Node(point));
                this->canNodeVec_.insert(newNode);
                this->roadmap_->insert(newNode);
                newNodes.push_back(newNode);
            }
        }
		
		// remove new nodes from uncovered global_points
		for (std::shared_ptr<PRM::Node>& n : newNodes) {
            this->uncoveredGlobalPoints_.erase(n->pos);

			n->newNode = true;
		}

        // node connection
        for (const std::shared_ptr<PRM::Node>& n : this->canNodeVec_) {
            n->adjNodes.clear();
			std::vector<std::shared_ptr<PRM::Node>> knn = this->roadmap_->kNearestNeighbor(n, this->nnNum_);

			for (std::shared_ptr<PRM::Node>& nearestNeighborNode : knn) { 
				bool hasCollision = not this->map_->isInflatedFreeLine(n->pos, nearestNeighborNode->pos);
                if (hasCollision == false) {
                    n->adjNodes.insert(nearestNeighborNode);
				}
			}
		}
	}	 

	void CAN::pruneNodes(){
		// record the invalid nodes
		std::unordered_set<std::shared_ptr<PRM::Node>> invalidSet;
		for (std::shared_ptr<PRM::Node> n : this->canNodeVec_){ // new nodes
			if (not this->isPosValid(n->pos, this->safeDistXY_, this->safeDistZ_)){// 1. new nodes
			// if (this->map_->isInflatedOccupied(n->pos)){// 1. new nodes
				invalidSet.insert(n);
			}	
		}

		// remove invalid nodes
		for (std::shared_ptr<PRM::Node> in : invalidSet){
			this->canNodeVec_.erase(in);
			this->roadmap_->remove(in);
            this->uncoveredGlobalPoints_.insert(in->pos);  // insert back to uncovered global points set
		}


		//  remove invalid edges
		for (std::shared_ptr<PRM::Node> n : this->canNodeVec_){
			std::vector<std::shared_ptr<PRM::Node>> eraseVec;
			for (std::shared_ptr<PRM::Node> neighbor : n->adjNodes){
				if (invalidSet.find(neighbor) != invalidSet.end()){
					eraseVec.push_back(neighbor);
				}
			}

			for (std::shared_ptr<PRM::Node> en : eraseVec){
				n->adjNodes.erase(en);
			}
		}
	}

	void CAN::updateInformationGain(){
		for (std::shared_ptr<PRM::Node> updateN : this->canNodeVec_){ // update information gain
			std::unordered_map<double, int> yawNumVoxels;
			int unknownVoxelNum = this->calculateUnknown(updateN, yawNumVoxels);
			updateN->numVoxels = unknownVoxelNum;
			updateN->yawNumVoxels = yawNumVoxels;
			updateN->newNode = false;
		}
		this->histTraj_.clear(); // clear history
	}


	void CAN::odomCB(const nav_msgs::OdometryConstPtr& odom){
		this->odom_ = *odom;
		this->position_ = Eigen::Vector3d (this->odom_.pose.pose.position.x, this->odom_.pose.pose.position.y, this->odom_.pose.pose.position.z);
		this->currYaw_ = globalPlanner::rpy_from_quaternion(this->odom_.pose.pose.orientation);
		this->odomReceived_ = true;

		if (this->histTraj_.size() == 0){
			this->histTraj_.push_back(this->position_);
		}
		else{
			Eigen::Vector3d lastPos = this->histTraj_.back();
			double dist = (this->position_ - lastPos).norm();
			if (dist >= 0.5){
				if (this->histTraj_.size() >= 100){
					this->histTraj_.pop_front();
					this->histTraj_.push_back(this->position_);
				}
				else{
					this->histTraj_.push_back(this->position_);
				}
			}
		}
	}

	void CAN::ultimateTargetCB(const geometry_msgs::PointStamped::ConstPtr& navTarget) {
		Eigen::Vector3d ultimateTargetPos(navTarget->point.x, navTarget->point.y, navTarget->point.z);

		this->ultimateTarget_.reset(new PRM::Node (ultimateTargetPos));

		cout << "[Roadmap]: Ultimate navigation target received. " << endl;
	}

	void CAN::visCB(const ros::TimerEvent&){
		if (this->canNodeVec_.size() != 0){
			visualization_msgs::MarkerArray roadmapMarkers = this->buildRoadmapMarkers();
			this->roadmapPub_.publish(roadmapMarkers);
		}

		if (this->bestPath_.size() != 0){
			this->publishBestPath();
		}
	}

	void CAN::currGoalCB(const geometry_msgs::PointStamped::ConstPtr& goal) {
		Eigen::Vector3d p(goal->point.x, goal->point.y, goal->point.z);
		this->currGoal_.reset(new PRM::Node(p));

		this->currGoalReceived_ = true;
		this->resettingRLEnv_ = false;
	}

	void CAN::waypointUpdateCB(const ros::TimerEvent&) {
        ROS_INFO("waypointUpdate callback called");
        try {
            if (this->resettingRLEnv_) {
                assert(this->currGoal_ != nullptr);
                // Publishing robot initial position as waypoint to reset RL Env
                geometry_msgs::PointStamped waypoint;
                waypoint.header.stamp = ros::Time::now();
                waypoint.header.frame_id = "map";
                waypoint.point.x = this->currGoal_->pos(0);
                waypoint.point.y = this->currGoal_->pos(1);
                waypoint.point.z = this->currGoal_->pos(2);
                this->waypointPub_.publish(waypoint);
                // std::cout << "\033[1;32m[Debug]: Publishing robot initial position as waypoint to reset RL Env... \033[0m" << std::endl;
            }
            else if (this->currGoalReceived_) {
                assert(this->currGoal_ != nullptr);
                assert(this->roadmap_ != nullptr);
                assert(this->map_ != nullptr);

                Point3D lastNavWaypoint = this->navWaypoint_;

                // find best path to current goal
                this->bestPath_.clear();
                // find nearest node of current location
                std::shared_ptr<PRM::Node> currPos(new PRM::Node (this->position_));
                std::shared_ptr<PRM::Node> start = this->roadmap_->nearestNeighbor(currPos);
                std::shared_ptr<PRM::Node> temp_goal = this->roadmap_->nearestNeighbor(this->currGoal_);  // this is important, or no path will be found!

                std::vector<std::shared_ptr<PRM::Node>> path = PRM::AStar(this->roadmap_, start, temp_goal, this->map_);
                if (int(path.size()) != 0) {
                    // Finding path success
                    path.insert(path.begin(), currPos);
                    std::vector<std::shared_ptr<PRM::Node>> pathSc;
                    this->shortcutPath(path, pathSc);
                    this->bestPath_ = pathSc;
                    this->waypointIdx_ = 0;
                }

                // construct waypoint
                geometry_msgs::PointStamped waypoint;
                waypoint.header.stamp = ros::Time::now();
                waypoint.header.frame_id = "map";
                if (this->bestPath_.size() != 0) {
                    if (((this->position_ - this->bestPath_[this->waypointIdx_]->pos).norm() <= 0.2) && (this->waypointIdx_ < (this->bestPath_.size() - 1))){
                        this->waypointIdx_ += 1;
                    }

                    this->navWaypoint_ = Point3D(this->bestPath_[this->waypointIdx_]->pos(0), this->bestPath_[this->waypointIdx_]->pos(1), this->bestPath_[this->waypointIdx_]->pos(2));

                    Point3D projectedWaypoint = this->projectNavWaypoint(this->navWaypoint_, lastNavWaypoint);

                    waypoint.point.x = projectedWaypoint.x;
                    waypoint.point.y = projectedWaypoint.y;
                    waypoint.point.z = projectedWaypoint.z;
                }
                else {
                    // Find global paths fails. Directly using local planner
                    // std::cout << "\033[1;31m[Debug]: Find global paths fails. Directly using local planner... \033[0m" << std::endl;
                    this->navHeading_ = Point3D(0, 0, 0);

                    waypoint.point.x = this->currGoal_->pos(0);
                    waypoint.point.y = this->currGoal_->pos(1);
                    waypoint.point.z = this->currGoal_->pos(2);
                }
                this->waypointPub_.publish(waypoint);
            }
        } catch (const std::exception &e) {
            ROS_ERROR("Exception caught: %s", e.what());
        }
	}

	Point3D CAN::projectNavWaypoint(const Point3D& nav_waypoint, const Point3D& last_waypoint) {
		const float kEpsilon = 1e-5;
		const bool is_momentum = (last_waypoint - nav_waypoint).norm() < kEpsilon ? true : false; // momentum heading if same goal
		Point3D waypoint = nav_waypoint;
		const float waypoint_project_dist = 1.6;  // 5.0 in far-planner
		const Point3D robot_pos(this->position_);
		const Point3D diff_p = nav_waypoint - robot_pos;
		Point3D new_heading;
		if (is_momentum && this->navHeading_.norm() > kEpsilon) {
			const float hdist = waypoint_project_dist / 2.0f;
			const float ratio = std::min(hdist, diff_p.norm()) / hdist;
			new_heading = diff_p.normalize() * ratio + this->navHeading_ * (1.0f - ratio);
		} else {
			new_heading = diff_p.normalize();
		}
		if (this->navHeading_.norm() > kEpsilon && new_heading.norm_dot(this->navHeading_) < 0.0f) { // negative direction reproject
			Point3D temp_heading(this->navHeading_.y, -this->navHeading_.x, this->navHeading_.z);
			if (temp_heading.norm_dot(new_heading) < 0.0f) {
			temp_heading.x = -temp_heading.x, temp_heading.y = -temp_heading.y;
			}
			new_heading = temp_heading;
		}
		this->navHeading_ = new_heading.normalize();
		if (diff_p.norm() < waypoint_project_dist) {
			waypoint = nav_waypoint + this->navHeading_ * (waypoint_project_dist - diff_p.norm());
		}
		return waypoint;
	}

	bool CAN::isPosValid(const Eigen::Vector3d& p){
		for (double x=p(0)-this->safeDistXY_; x<=p(0)+this->safeDistXY_; x+=this->map_->getRes()){
			for (double y=p(1)-this->safeDistXY_; y<=p(1)+this->safeDistXY_; y+=this->map_->getRes()){
				for (double z=p(2)-this->safeDistZ_; z<=p(2)+this->safeDistZ_; z+=this->map_->getRes()){
					if (this->safeDistCheckUnknown_){
						if (not this->map_->isInflatedFree(Eigen::Vector3d (x, y, z))){
							return false;
						}
					}
					else{
						if (this->map_->isInflatedOccupied(Eigen::Vector3d (x, y, z))){
							return false;
						}
					}
				}
			}
		}
		return true;			
	}


	bool CAN::isPosValid(const Eigen::Vector3d& p, double safeDistXY, double safeDistZ){
		for (double x=p(0)-safeDistXY; x<=p(0)+safeDistXY; x+=this->map_->getRes()){
			for (double y=p(1)-safeDistXY; y<=p(1)+safeDistXY; y+=this->map_->getRes()){
				for (double z=p(2)-safeDistZ; z<=p(2)+safeDistZ; z+=this->map_->getRes()){
					if (this->safeDistCheckUnknown_){
						if (not this->map_->isInflatedFree(Eigen::Vector3d (x, y, z))){
							return false;
						}
					}
					else{
						if (this->map_->isInflatedOccupied(Eigen::Vector3d (x, y, z))){
							return false;
						}
					}
				}
			}
		}
		return true;		
	}

	int CAN::calculateUnknown(const shared_ptr<PRM::Node>& n, std::unordered_map<double, int>& yawNumVoxels){
		for (double yaw : this->yaws_){
			yawNumVoxels[yaw] = 0;
		}
		// Position:
		Eigen::Vector3d p = n->pos;

		double zRange = this->dmax_ * tan(this->verticalFOV_/2.0);
		int countTotalUnknown = 0;
		for (double z = p(2) - zRange; z <= p(2) + zRange; z += this->map_->getRes()){
			for (double y = p(1) - this->dmax_; y <= p(1)+ this->dmax_; y += this->map_->getRes()){
				for (double x = p(0) - this->dmax_; x <= p(0) + this->dmax_; x += this->map_->getRes()){
					Eigen::Vector3d nodePoint (x, y, z);
					if (nodePoint(0) < this->globalRegionMin_(0) or nodePoint(0) > this->globalRegionMax_(0) or
						nodePoint(1) < this->globalRegionMin_(1) or nodePoint(1) > this->globalRegionMax_(1) or
						nodePoint(2) < this->globalRegionMin_(2) or nodePoint(2) > this->globalRegionMax_(2)){
						// not in global range
						continue;
					}

					if (this->map_->isUnknown(nodePoint) and not this->map_->isInflatedOccupied(nodePoint)){
						if (this->sensorFOVCondition(nodePoint, p)){
							++countTotalUnknown;
							for (double yaw: this->yaws_){
								Eigen::Vector3d yawDirection (cos(yaw), sin(yaw), 0);
								Eigen::Vector3d direction = nodePoint - p;
								Eigen::Vector3d face (direction(0), direction(1), 0);
								double angleToYaw = angleBetweenVectors(face, yawDirection);
								if (angleToYaw <= this->horizontalFOV_/2){
									yawNumVoxels[yaw] += 1;
								}
							}
						}
					}
				}
			}
		}
		return countTotalUnknown;
	}

	double CAN::calculatePathLength(const std::vector<shared_ptr<PRM::Node>>& path){
		int idx1 = 0;
		double length = 0;
		for (size_t idx2=1; idx2<=path.size()-1; ++idx2){
			length += (path[idx2]->pos - path[idx1]->pos).norm();
			++idx1;
		}
		return length;
	}

	void CAN::shortcutPath(const std::vector<std::shared_ptr<PRM::Node>>& path, std::vector<std::shared_ptr<PRM::Node>>& pathSc){
		size_t ptr1 = 0; size_t ptr2 = 2;
		pathSc.push_back(path[ptr1]);

		if (path.size() == 1){
			return;
		}

		if (path.size() == 2){
			pathSc.push_back(path[1]);
			return;
		}

		while (ros::ok()){
			if (ptr2 > path.size()-1){
				break;
			}
			std::shared_ptr<PRM::Node> p1 = path[ptr1];
			std::shared_ptr<PRM::Node> p2 = path[ptr2];
			Eigen::Vector3d pos1 = p1->pos;
			Eigen::Vector3d pos2 = p2->pos;
			bool lineValidCheck;
			// lineValidCheck = not this->map_->isInflatedOccupiedLine(pos1, pos2);
			lineValidCheck = this->map_->isInflatedFreeLine(pos1, pos2);
			// double maxDistance = std::numeric_limits<double>::max();
			// double maxDistance = 3.0;
			// if (lineValidCheck and (pos1 - pos2).norm() <= maxDistance){
			if (lineValidCheck){
				if (ptr2 == path.size()-1){
					pathSc.push_back(p2);
					break;
				}
				++ptr2;
			}
			else{
				pathSc.push_back(path[ptr2-1]);
				if (ptr2 == path.size()-1){
					pathSc.push_back(p2);
					break;
				}
				ptr1 = ptr2-1;
				ptr2 = ptr1+2;
			}
		}		
	}

	visualization_msgs::MarkerArray CAN::buildRoadmapMarkers(){  
		visualization_msgs::MarkerArray roadmapMarkers;

		// PRM nodes and edges
		int countPointNum = 0;
		int countEdgeNum = 0;
		int countVoxelNumText = 0;
		for (std::shared_ptr<PRM::Node> n : this->canNodeVec_){
			// Node point
			visualization_msgs::Marker point;
			point.header.frame_id = "map";
			point.header.stamp = ros::Time::now();
			point.ns = "prm_point";
			point.id = countPointNum;
			point.type = visualization_msgs::Marker::SPHERE;
			point.action = visualization_msgs::Marker::ADD;
			point.pose.position.x = n->pos(0);
			point.pose.position.y = n->pos(1);
			point.pose.position.z = n->pos(2);
			point.lifetime = ros::Duration(0.2); //5
			point.scale.x = 0.05;
			point.scale.y = 0.05;
			point.scale.z = 0.05;
			point.color.a = 1.0;
			point.color.r = 1.0;
			point.color.g = 1.0;
			point.color.b = 0.0;
			++countPointNum;
			roadmapMarkers.markers.push_back(point);

			// number of voxels for each node
			visualization_msgs::Marker voxelNumText;
			voxelNumText.ns = "num_voxel_text";
			voxelNumText.header.frame_id = "map";
			voxelNumText.id = countVoxelNumText;
			voxelNumText.header.stamp = ros::Time::now();
			voxelNumText.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
			voxelNumText.action = visualization_msgs::Marker::ADD;
			voxelNumText.pose.position.x = n->pos(0);
			voxelNumText.pose.position.y = n->pos(1);
			voxelNumText.pose.position.z = n->pos(2)+0.1;
			voxelNumText.scale.x = 0.2;
			voxelNumText.scale.y = 0.2;
			voxelNumText.scale.z = 0.2;
			voxelNumText.color.a = 1.0;
			voxelNumText.text = std::to_string(n->numVoxels);
			voxelNumText.lifetime = ros::Duration(0.2); //5
			++countVoxelNumText;
			roadmapMarkers.markers.push_back(voxelNumText);

			// Edges 
			visualization_msgs::Marker line;
			line.ns = "edge";
			line.header.frame_id = "map";
			line.type = visualization_msgs::Marker::LINE_LIST;
			line.header.stamp = ros::Time::now();

			for (std::shared_ptr<PRM::Node> adjNode : n->adjNodes){
				geometry_msgs::Point p1, p2;
				p1.x = n->pos(0);
				p1.y = n->pos(1);
				p1.z = n->pos(2);
				p2.x = adjNode->pos(0);
				p2.y = adjNode->pos(1);
				p2.z = adjNode->pos(2);				
				line.points.push_back(p1);
				line.points.push_back(p2);
				line.id = countEdgeNum;
				line.scale.x = 0.02;
				line.scale.y = 0.02;
				line.scale.z = 0.02;
				line.color.r = 1.0;
				line.color.g = 1.0;
				line.color.b = 0.0;
				line.color.a = 1.0;
				line.lifetime = ros::Duration(0.2); //5
				++countEdgeNum;
				// roadmapMarkers.markers.push_back(line);
			}
			if(n->adjNodes.empty()) {
				geometry_msgs::Point p1;
				p1.x = n->pos(0);
				p1.y = n->pos(1);
				p1.z = n->pos(2);
				line.points.push_back(p1);
			}
			roadmapMarkers.markers.push_back(line);
		}

		return roadmapMarkers;
	}

	void CAN::publishBestPath(){
		visualization_msgs::MarkerArray bestPathMarkers;
		int countNodeNum = 0;
		int countLineNum = 0;
		for (size_t i=0; i<this->bestPath_.size(); ++i){
			std::shared_ptr<PRM::Node> n = this->bestPath_[i];
			visualization_msgs::Marker point;
			point.header.frame_id = "map";
			point.header.stamp = ros::Time::now();
			point.ns = "best_path_node";
			point.id = countNodeNum;
			point.type = visualization_msgs::Marker::SPHERE;
			point.action = visualization_msgs::Marker::ADD;
			point.pose.position.x = n->pos(0);
			point.pose.position.y = n->pos(1);
			point.pose.position.z = n->pos(2);
			point.lifetime = ros::Duration(0.5);
			point.scale.x = 0.01;
			point.scale.y = 0.01;
			point.scale.z = 0.01;
			point.color.a = 1.0;
			point.color.r = 1.0;
			point.color.g = 1.0;
			point.color.b = 1.0;
			++countNodeNum;
			bestPathMarkers.markers.push_back(point);

			if (i<this->bestPath_.size()-1){
				std::shared_ptr<PRM::Node> nNext = this->bestPath_[i+1];
				visualization_msgs::Marker line;
				line.ns = "best_path";
				line.header.frame_id = "map";
				line.type = visualization_msgs::Marker::LINE_LIST;
				line.header.stamp = ros::Time::now();
				geometry_msgs::Point p1, p2;
				p1.x = n->pos(0);
				p1.y = n->pos(1);
				p1.z = n->pos(2);
				p2.x = nNext->pos(0);
				p2.y = nNext->pos(1);
				p2.z = nNext->pos(2);				
				line.points.push_back(p1);
				line.points.push_back(p2);
				line.id = countLineNum;
				line.scale.x = 0.1;
				line.scale.y = 0.1;
				line.scale.z = 0.1;
				line.color.r = 1.0;
				line.color.g = 0.0;
				line.color.b = 0.0;
				line.color.a = 1.0;
				line.lifetime = ros::Duration(0.5);
				++countLineNum;
				bestPathMarkers.markers.push_back(line);				
			}
		}
		this->bestPathPub_.publish(bestPathMarkers);		
	}

}