import pygraph
from autolab_core import RigidTransform

class TFGraph(pygraph.DirectedGraph):
    def new_tf_node(self, robot_id):
        node_id = self.new_node()
        self.nodes[node_id]['data']['robot_id'] = robot_id

    def get_tf_node(self, robot_id):
        for key,node in self.nodes.items():
            if node['data']['robot_id'] == robot_id:
                return key,node
        raise NonexistentNodeError(robot_id)
        return None,None

    def new_tf_edge(self, from_robot_id, to_robot_id, Rigidtrans):
        from_node_id,from_node = self.get_tf_node(from_robot_id)
        to_node_id,to_node = self.get_tf_node(to_robot_id)
        edge_id = self.new_edge(from_node_id, to_node_id, 1)
        self.edges[edge_id]['data']['tf'] = Rigidtrans
        edge_id = self.new_edge(to_node_id, from_node_id, 1)
        self.edges[edge_id]['data']['tf'] = Rigidtrans.inverse()

    def get_graph_edge(self, from_node_id, to_node_id):
        for edge_id in self.nodes[from_node_id]['edges']:
            if self.edges[edge_id]['vertices'][1] == to_node_id:
                return edge_id,self.edges[edge_id]
    
    def get_tf_edge(self, from_robot_id, to_robot_id):
        from_node_id,from_node = self.get_tf_node(from_robot_id)
        to_node_id,to_node = self.get_tf_node(to_robot_id)
        return get_graph_edge(from_node_id, to_node_id)

    def get_tf(self, from_robot_id, to_robot_id):
        from_node_id,from_node = self.get_tf_node(from_robot_id)
        to_node_id,to_node = self.get_tf_node(to_robot_id)
        path = pygraph.a_star_search(self, from_node_id, to_node_id)
        if path == []:
            return None
        elif len(path) == 1:
            return RigidTransform()
        result = self.get_graph_edge(path[0], path[1])[1]['data']['tf']
        for i in range(1,len(path)-1):
            result = self.get_graph_edge(path[i], path[i+1])[1]['data']['tf'] * result
        return result

    def get_connected_robots(self, robot_id):
        node_id,node = self.get_tf_node(robot_id)
        components = pygraph.get_connected_components(self)
        result_node_id = []
        for node_list in components:
            if node_id in node_list:
                result_node_id = node_list
        result_robot_id = []
        for con_node_id in result_node_id:
            result_robot_id.append(self.nodes[con_node_id]['data']['robot_id'])
        return result_robot_id

if __name__ == '__main__':
    tf_graph = TFGraph()
    tf_graph.new_tf_node(1)
    print pygraph.graph_to_dot(tf_graph)
    tf_graph.new_tf_node(3)
    print pygraph.graph_to_dot(tf_graph)
    tf_graph.new_tf_node(2)
    print pygraph.graph_to_dot(tf_graph)
    tf_graph.new_tf_node(5)
    print pygraph.graph_to_dot(tf_graph)
    tf = RigidTransform(from_frame = '1', to_frame = '2')
    tf_graph.new_tf_edge(1,2,tf)
    print pygraph.graph_to_dot(tf_graph)
    tf = RigidTransform(from_frame = '2', to_frame = '3')
    tf_graph.new_tf_edge(2,3,tf)
    print pygraph.graph_to_dot(tf_graph)
    tf = RigidTransform(from_frame = '3', to_frame = '5')
    tf_graph.new_tf_edge(3,5,tf)
    print pygraph.graph_to_dot(tf_graph)
    tf = tf_graph.get_tf(1,5)
    print tf
    import pdb; pdb.set_trace()