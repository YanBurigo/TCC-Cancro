from flask_restx import Api, Resource, fields

rest_api = Api(version="1.0", title="Neural Network API", default="Neural Network", default_label="Convolutional Analysis")

# Used to validate input data for creation
create_model = rest_api.model('RequestAnalysisCommand', {"data": fields.String(required=True, min_length=1, max_length=255)})


"""
    Flask-Restx routes
"""

@rest_api.route('/api/request-analysis', doc={"description": 'Analysis Images In Convolutional Neural Network'})
class Items(Resource):

    @rest_api.expect(create_model, validate=True)
    def post(self):
        """
            Returns if a plant is infected with Canker
        """
        #Code...
        
        return {"success": True}, 200
