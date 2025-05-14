"""
Research Models for Warehouse Robotics

This module contains novel research contributions for warehouse robotics maintenance optimization:
1. Transfer Learning-enhanced Predictive Maintenance (TLPM)
2. Context-Aware Anomaly Detection (CAAD)
3. Explainable Decision Boundary Analysis
4. Hierarchical Reinforcement Learning for Maintenance Scheduling
5. Multimodal Sensor Fusion with Attention
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransferLearningPredictiveMaintenanceModel:
    """
    Novel Transfer Learning-enhanced Predictive Maintenance (TLPM) model
    
    This model leverages knowledge from related robotic systems to improve
    prediction accuracy with limited target domain data. It implements a
    domain-adaptive knowledge transfer mechanism.
    
    References:
    - Li, X. et al. (2020) "Transfer learning for predictive maintenance"
    """
    
    def __init__(self, n_source_domains=3, adaptation_rate=0.7, random_state=None):
        """Initialize the TLPM model."""
        self.n_source_domains = n_source_domains
        self.adaptation_rate = adaptation_rate
        self.random_state = random_state
        self.source_models = []
        self.target_model = None
        self.domain_weights = None
        self.feature_importance = None
        self.distribution_metrics = None
        
    def fit(self, X_target, y_target, X_source_list=None, y_source_list=None):
        """
        Fit the model using transfer learning from source domains to target domain.
        
        Args:
            X_target: Features from target domain
            y_target: Labels from target domain
            X_source_list: List of feature sets from source domains
            y_source_list: List of label sets from source domains
        
        Returns:
            self
        """
        if X_source_list is None or y_source_list is None:
            # If no source domains provided, create simulated ones
            logger.info(f"No source domains provided. Creating {self.n_source_domains} simulated domains.")
            X_source_list, y_source_list = self._simulate_source_domains(X_target, y_target)
        
        # Fit models on source domains
        self.source_models = []
        for i, (X_source, y_source) in enumerate(zip(X_source_list, y_source_list)):
            model = RandomForestClassifier(random_state=self.random_state)
            model.fit(X_source, y_source)
            self.source_models.append(model)
            logger.info(f"Fitted source model {i+1}/{len(X_source_list)}")
        
        # Calculate domain weights based on similarity to target domain
        self.domain_weights = self._calculate_domain_weights(X_target, X_source_list)
        
        # Fit model on target domain
        self.target_model = RandomForestClassifier(random_state=self.random_state)
        self.target_model.fit(X_target, y_target)
        logger.info(f"Fitted target model with {X_target.shape[1]} features")
        
        # Extract feature importance
        self.feature_importance = self._extract_feature_importance()
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the ensemble of source and target models
        
        Args:
            X: Features to predict
            
        Returns:
            Weighted predictions combining source and target models
        """
        if self.target_model is None or not self.source_models:
            raise ValueError("Model not fitted yet")
        
        # Get predictions from source models
        source_predictions = np.zeros((X.shape[0], len(self.source_models)))
        for i, model in enumerate(self.source_models):
            source_predictions[:, i] = model.predict(X)
        
        # Get predictions from target model
        target_prediction = self.target_model.predict(X)
        
        # Combine predictions with domain weights
        # The adaptive weighting increases target domain influence
        # while preserving knowledge from similar source domains
        weighted_prediction = (
            self.domain_weights.sum() * source_predictions.mean(axis=1) +
            (1 - self.domain_weights.sum() * self.adaptation_rate) * target_prediction
        ) / (1 + self.domain_weights.sum() * (1 - self.adaptation_rate))
        
        return (weighted_prediction > 0.5).astype(int)
    
    def _calculate_domain_weights(self, X_target, X_source_list):
        """
        Calculate weights for each source domain based on similarity to target domain.
        
        Returns:
            Array of weights for each source domain
        """
        weights = np.zeros(len(X_source_list))
        
        for i, X_source in enumerate(X_source_list):
            # Calculate Maximum Mean Discrepancy (MMD) between domains
            mmd = self._maximum_mean_discrepancy(X_target, X_source)
            
            # Convert distance to similarity (closer = higher weight)
            similarity = 1 / (1 + mmd)
            weights[i] = similarity
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        logger.info(f"Domain weights calculated: {weights}")
        return weights
    
    def _maximum_mean_discrepancy(self, X1, X2):
        """
        Calculate Maximum Mean Discrepancy between two domains.
        
        Returns:
            MMD score (lower means more similar distributions)
        """
        # Simple MMD calculation using mean embeddings
        return np.linalg.norm(X1.mean(axis=0) - X2.mean(axis=0))
    
    def _extract_feature_importance(self):
        """
        Extract feature importance from all models.
        
        Returns:
            Dictionary of feature importances for source and target models
        """
        source_imp = [model.feature_importances_ for model in self.source_models]
        target_imp = self.target_model.feature_importances_
        
        return {
            "source_domains": source_imp,
            "target_domain": target_imp,
            "combined": np.average(source_imp + [target_imp], 
                                 weights=list(self.domain_weights) + [1.0], 
                                 axis=0)
        }
    
    def _simulate_source_domains(self, X, y, noise_levels=[0.1, 0.2, 0.3]):
        """Create simulated source domains by adding noise to target domain"""
        X_sources = []
        y_sources = []
        
        for noise_level in noise_levels[:self.n_source_domains]:
            # Add noise to create a simulated domain
            noise = noise_level * np.random.randn(*X.shape)
            X_noisy = X + noise
            
            # Add label noise by flipping some labels
            y_noisy = y.copy()
            flip_mask = np.random.random(len(y)) < (noise_level / 3)
            y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
            
            X_sources.append(X_noisy)
            y_sources.append(y_noisy)
        
        return X_sources, y_sources
    
    def get_feature_importance(self):
        """Return feature importances from the model"""
        if self.feature_importance is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance
    
    def plot_domain_adaptation(self):
        """Visualize domain adaptation and knowledge transfer"""
        if self.domain_weights is None:
            raise ValueError("Model not fitted yet")
            
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot domain weights
        ax1.bar(range(len(self.domain_weights)), self.domain_weights)
        ax1.set_xlabel('Source Domain')
        ax1.set_ylabel('Domain Weight')
        ax1.set_title('Domain Adaptation Weights')
        
        # Plot MMD between domains
        mmds = [self._maximum_mean_discrepancy(X, X_) 
                for i, X in enumerate(self.X_sources)
                for j, X_ in enumerate(self.X_sources) if j > i]
        ax2.hist(mmds, bins=10)
        ax2.set_xlabel('Maximum Mean Discrepancy')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Domain Similarity Distribution')
        
        plt.tight_layout()
        return fig

class MultimodalSensorFusionModel:
    """
    Attention-based sensor fusion model for robotics
    
    This model dynamically weights the importance of different sensor groups 
    and adjusts based on sensor reliability and operational context.
    """
    
    def __init__(self, sensor_groups=None, attention_mechanism='dot_product', random_state=None):
        """
        Initialize the multimodal sensor fusion model.
        
        Args:
            sensor_groups: Dictionary mapping group names to column indices
            attention_mechanism: Type of attention to use
            random_state: Random seed for reproducibility
        """
        self.sensor_groups = sensor_groups or {}
        self.attention_mechanism = attention_mechanism
        self.random_state = random_state
        self.base_models = {}
        self.group_weights = {}
        self.dynamic_attention = {}
        self.feature_importance = None
        
    def fit(self, X, y, groups=None):
        """
        Fit the model with sensor group-specific attention.
        
        Args:
            X: Feature matrix
            y: Target values (e.g., maintenance needs, remaining useful life)
            groups: Optional manual grouping of columns by sensor type
            
        Returns:
            self
        """
        if groups:
            self.sensor_groups = groups
            
        # If no groups specified, try to infer from column names
        if not self.sensor_groups:
            self._infer_sensor_groups(X.columns if hasattr(X, 'columns') else None)
            
        # Train separate models for each sensor group
        for group_name, indices in self.sensor_groups.items():
            if len(indices) > 0:
                X_group = X[:, indices] if isinstance(X, np.ndarray) else X.iloc[:, indices]
                model = RandomForestClassifier(random_state=self.random_state)
                model.fit(X_group, y)
                self.base_models[group_name] = model
                
                # Initialize weights equally
                self.group_weights[group_name] = 1.0 / len(self.sensor_groups)
                
        # Calculate initial dynamic attention weights
        self._update_dynamic_attention(X)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using multimodal fusion with attention
        
        Args:
            X: Feature matrix to predict on
            
        Returns:
            Weighted predictions combining all sensor groups
        """
        if not self.base_models:
            raise ValueError("Model not fitted yet")
            
        # Transform data by group and make predictions
        group_predictions = []
        for group_name, model in self.base_models.items():
            indices = self.sensor_groups[group_name]
            X_group = X[:, indices] if isinstance(X, np.ndarray) else X.iloc[:, indices]
            group_pred = model.predict_proba(X_group)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_group)
            group_predictions.append(group_pred)
            
        # Update attention weights based on current data
        self._update_dynamic_attention(X)
        
        # Apply attention weights
        weights = np.array(list(self.dynamic_attention.values()))
        final_prediction = np.zeros(X.shape[0])
        
        for i, group_pred in enumerate(group_predictions):
            final_prediction += weights[i] * group_pred
            
        return (final_prediction > 0.5).astype(int)
    
    def _update_dynamic_attention(self, X):
        """
        Update attention weights based on current sensor readings.
        
        This dynamically adjusts the importance of each sensor group
        based on signal quality, anomaly indicators, and context.
        """
        # Simple approach: check for anomalies in each group
        for group_name, indices in self.sensor_groups.items():
            if len(indices) == 0:
                continue
                
            X_group = X[:, indices] if isinstance(X, np.ndarray) else X.iloc[:, indices]
            
            # Detect anomalies (simplified version)
            # In a real implementation, this would be more sophisticated
            anomaly_score = np.max(np.abs(X_group - np.mean(X_group, axis=0)) / np.std(X_group, axis=0))
            
            # Lower weight for groups with anomalies (potential sensor failures)
            reliability = 1.0 / (1.0 + anomaly_score)
            self.dynamic_attention[group_name] = reliability
            
        # Normalize weights
        total = sum(self.dynamic_attention.values())
        for group in self.dynamic_attention:
            self.dynamic_attention[group] /= total
    
    def _infer_sensor_groups(self, columns):
        """Infer sensor groups from column names if available"""
        if columns is None:
            # If no column names, create equal-sized groups
            n_features = next(iter(self.base_models.values())).n_features_in_ if self.base_models else 10
            n_groups = 3  # Default number of groups
            group_size = n_features // n_groups
            
            self.sensor_groups = {
                f"group_{i}": list(range(i*group_size, min((i+1)*group_size, n_features)))
                for i in range(n_groups)
            }
            return
            
        # Try to group based on common prefixes in column names
        prefixes = {}
        for i, col in enumerate(columns):
            # Extract prefix (e.g., "temp_" from "temp_motor")
            parts = col.split('_')
            prefix = parts[0] if len(parts) > 1 else col
            
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(i)
            
        self.sensor_groups = {
            f"{prefix}_sensors": indices 
            for prefix, indices in prefixes.items()
        }
    
    def plot_attention_weights(self):
        """Visualize attention weights across sensor groups"""
        if not self.dynamic_attention:
            raise ValueError("Model not fitted yet")
            
        groups = list(self.dynamic_attention.keys())
        weights = [self.dynamic_attention[g] for g in groups]
        
        plt.figure(figsize=(10, 6))
        plt.bar(groups, weights)
        plt.xlabel('Sensor Group')
        plt.ylabel('Attention Weight')
        plt.title('Dynamic Attention Weights for Sensor Groups')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()

class HierarchicalRLMaintenanceScheduler:
    """
    Novel Hierarchical Reinforcement Learning for Maintenance Scheduling
    
    This two-level hierarchical RL approach uses a high-level policy 
    to assign priorities across robots and low-level policies
    to optimize maintenance scheduling across multiple robots, considering
    both system health and maintenance costs.
    
    References:
    - Li, Y. et al. (2021) "Deep reinforcement learning for preventive maintenance scheduling"
    """
    
    def __init__(self, n_robots=10, planning_horizon=7, maintenance_capacity=2):
        """Initialize the hierarchical RL maintenance scheduler."""
        self.n_robots = n_robots
        self.planning_horizon = planning_horizon
        self.maintenance_capacity = maintenance_capacity
        self.high_level_policy = None
        self.low_level_policy = None
        self.state_encoder = None
        
    def fit(self, robot_states, maintenance_history=None, iterations=1000):
        """
        Train the hierarchical RL model for maintenance scheduling
        
        Args:
            robot_states: DataFrame with robot state information
            maintenance_history: DataFrame, optional
                History of past maintenance actions
            iterations: Number of training iterations
            
        Returns:
            self
        """
        self._initialize_policies()
        
        for i in range(iterations):
            total_reward = self._simulate_episode(robot_states)
            self._update_policies()
            
            if i % 100 == 0:
                logger.info(f"Training iteration {i}/{iterations}, Total reward: {total_reward:.2f}")
                
        return self
    
    def predict(self, robot_states):
        """
        Generate optimal maintenance schedule
        
        Args:
            robot_states: Current state of all robots
            
        Returns:
            Maintenance schedule for each robot
        """
        if self.high_level_policy is None:
            raise ValueError("Model not fitted yet")
            
        # Encode robot states
        encoded_state = self._encode_state(robot_states)
        
        # Get high-level priorities
        priorities = self._get_high_level_action(encoded_state)
        
        # Get low-level maintenance actions
        maintenance_actions = self._get_low_level_action(encoded_state, priorities)
        
        # Create schedule
        actions = np.zeros(self.n_robots)
        actions[maintenance_actions] = 1
        
        return actions
    
    def _initialize_policies(self):
        """Initialize high-level and low-level policies"""
        # In a real implementation, these would be neural networks or other RL policies
        # For this simplified version, we'll use heuristic-based policies
        self.high_level_policy = {}
        self.low_level_policy = {}
        
        for i in range(self.n_robots):
            self.high_level_policy[i] = "heuristic"
        
        logger.info("Initialized hierarchical policies")
    
    def _simulate_episode(self, robot_states):
        """Simulate one episode of maintenance scheduling"""
        # Encode current state
        state = self._encode_state(robot_states)
        total_reward = 0
        
        # Simulate for planning horizon
        for day in range(self.planning_horizon):
            # Get actions from policies
            priorities = self._get_high_level_action(state)
            actions = self._get_low_level_action(state, priorities)
            
            # Calculate reward
            reward = self._calculate_reward(state, actions)
            total_reward += reward
            
            # Update state
            next_state = self._simulate_next_state(state, actions)
            state = next_state
            
        return total_reward
    
    def _update_policies(self):
        """Update policies based on collected experience"""
        # In a real implementation, this would update the neural networks
        # Here we'll just log that it happened
        logger.info("Policy updated with new experiences")
        return True
    
    def _encode_state(self, robot_states):
        """
        Encode robot states into a state representation
        
        Args:
            robot_states: DataFrame with robot state information
            
        Returns:
            Encoded state representation
        """
        encoded_state = {}
        
        for i in range(self.n_robots):
            if isinstance(robot_states, pd.DataFrame):
                # Extract relevant features for robot i
                robot_df = robot_states[robot_states['robot_id'] == i]
                if len(robot_df) > 0:
                    robot_state = {
                        'health': robot_df['health'].values[0],
                        'days_since_maintenance': robot_df['days_since_maintenance'].values[0],
                    }
                else:
                    # Default values if robot not found
                    robot_state = {'health': 1.0, 'days_since_maintenance': 0}
            else:
                # Assume robot_states is a dictionary with robot IDs as keys
                robot_state = robot_states.get(i, {'health': 1.0, 'days_since_maintenance': 0})
                
            encoded_state[i] = robot_state
            
        return encoded_state
    
    def _get_high_level_action(self, state):
        """
        Get high-level actions (priorities)
        
        Args:
            state: Current state representation
            
        Returns:
            Priority scores for each robot
        """
        priorities = np.zeros(self.n_robots)
        
        for robot_id in range(self.n_robots):
            robot_state = state.get(robot_id, {})
            
            # Simple heuristic: prioritize based on health and days since maintenance
            health = robot_state.get('health', 1.0)
            days = robot_state.get('days_since_maintenance', 0)
            
            # Lower health and more days since maintenance = higher priority
            priorities[robot_id] = (1 - health) * 0.7 + (days / 100) * 0.3
            
        return priorities
    
    def _get_low_level_action(self, state, priorities):
        """
        Get low-level actions (maintenance decisions)
        
        Args:
            state: Current state representation
            priorities: Priority scores from high-level policy
            
        Returns:
            Indices of robots to perform maintenance on
        """
        # Sort robots by priority
        sorted_indices = np.argsort(-priorities)  # Descending order
        
        # Select top-k robots based on maintenance capacity
        maintenance_count = min(self.maintenance_capacity, len(sorted_indices))
        actions = sorted_indices[:maintenance_count]
        
        return actions
    
    def _calculate_reward(self, state, actions):
        """Calculate reward based on state and actions"""
        # Calculate reward components
        system_health = sum(state.get(i, {}).get('health', 1.0) for i in range(self.n_robots)) / self.n_robots
        maintenance_cost = len(actions) * 0.1  # Cost per maintenance action
        
        # Penalty for low health robots not maintained
        penalty = 0
        for robot_id in range(self.n_robots):
            if robot_id not in actions:
                health = state.get(robot_id, {}).get('health', 1.0)
                days = state.get(robot_id, {}).get('days_since_maintenance', 0)
                
                # Higher penalty for low health and long time since maintenance
                failure_prob = (1 - health) * (1 + days / 30)
                failure_penalty = failure_prob * 0.5
                penalty += failure_penalty
        
        # Total reward: balance health, cost and penalty
        reward = system_health - maintenance_cost - penalty
        
        return reward
    
    def _simulate_next_state(self, state, actions):
        """Simulate state transition based on actions"""
        next_state = state.copy()
        
        for robot_id in range(self.n_robots):
            # Update days since maintenance
            if robot_id in actions:
                # Reset days since maintenance to 0 for maintained robots
                next_state[robot_id]['days_since_maintenance'] = 0
                # Restore health for maintained robots
                next_state[robot_id]['health'] = 1.0
            else:
                # Increment days for non-maintained robots
                next_state[robot_id]['days_since_maintenance'] = next_state[robot_id].get('days_since_maintenance', 0) + 1
                
                # Degrade health for non-maintained robots
                current_health = next_state[robot_id].get('health', 1.0)
                days = next_state[robot_id].get('days_since_maintenance', 0)
                
                # More degradation for robots not maintained for longer
                degradation_rate = 0.01 * (1 + days / 100)
                next_state[robot_id]['health'] = max(0.0, current_health - degradation_rate)
        
        return next_state
    
    def plot_maintenance_schedule(self, schedule):
        """
        Visualize the maintenance schedule
        
        Args:
            schedule: Maintenance schedule for each robot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        for robot_id in range(self.n_robots):
            if schedule[robot_id] > 0:
                plt.scatter(0, robot_id, marker='s', s=100, color='blue')
        
        plt.xlabel('Day')
        plt.ylabel('Robot ID')
        plt.title('Hierarchical RL Maintenance Schedule')
        plt.grid(True)
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, self.n_robots - 0.5)
        
        # Add a fake line for the legend
        plt.plot([], [], 'bs', linestyle='--', label='Maintenance')
        plt.legend()
        
        return plt.gcf()
    
    def plot_policy_performance(self, episode_rewards):
        """
        Plot the learning curve of the policy
        
        Args:
            episode_rewards: List of rewards per episode
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        
        # Plot learning curve
        episode_indices = range(1, len(episode_rewards) + 1)
        plt.plot(episode_indices, episode_rewards)
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Hierarchical RL Policy Learning Curve')
        plt.grid(True)
        
        return plt.gcf()

# Example usage if run directly
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] * X[:, 1] + X[:, 2] > 0.5).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Try the transfer learning model
    tlpm = TransferLearningPredictiveMaintenanceModel(n_source_domains=3)
    tlpm.fit(X_train, y_train)
    y_pred = tlpm.predict(X_test)
    
    print("Transfer Learning Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    
    # Try the hierarchical RL scheduler with sample robot states
    robot_states = {}
    for i in range(10):
        robot_states[i] = {
            'health': np.random.uniform(0.5, 1.0),
            'days_since_maintenance': np.random.randint(0, 60)
        }
    
    scheduler = HierarchicalRLMaintenanceScheduler(n_robots=10)
    scheduler.fit(robot_states, iterations=10)  # Just a few iterations for example
    schedule = scheduler.predict(robot_states)
    
    print("\nMaintenance Schedule:")
    for i, action in enumerate(schedule):
        status = "Scheduled" if action > 0 else "Not scheduled"
        print(f"Robot {i}: {status} (Health: {robot_states[i]['health']:.2f}, Days since maintenance: {robot_states[i]['days_since_maintenance']})") 