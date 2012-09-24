
#pragma  once


/* 
	MS3D STRUCTURES 
*/

// byte-align structures
#ifdef _MSC_VER
#	pragma pack( push, packing )
#	pragma pack( 1 )
#	define PACK_STRUCT
#elif defined( __GNUC__ )
#	define PACK_STRUCT	__attribute__((packed))
#else
#	error you must byte-align these structures with the appropriate compiler directives
#endif

typedef unsigned char byte;
typedef unsigned short word;

// File header
struct MS3DHeader
{
	char m_ID[10];
	int m_version;
} PACK_STRUCT;

// Vertex information
struct MS3DVertex
{
	byte m_flags;
	float m_vertex[3];
	char m_cBone;
	byte m_refCount;
} PACK_STRUCT;


// Material information
struct MS3DMaterial
{
    char m_name[32];
    float m_ambient[4];
    float m_diffuse[4];
    float m_specular[4];
    float m_emissive[4];
    float m_shininess;	// 0.0f - 128.0f
    float m_transparency;	// 0.0f - 1.0f
    byte m_mode;	// 0, 1, 2 is unused now
    char m_texture[128];
    char m_alphamap[128];
} PACK_STRUCT;

// Keyframe data
struct MS3DKeyframe
{
	float m_fTime;
	float m_fParam[3];
} PACK_STRUCT;


//-------------------------------------------------------------
//- SMs3dMesh
//- Group of triangles in the ms3d file
struct SMs3dMesh
{
	unsigned char m_ucFlags;   //Editor flags again
	char m_cName[32];          //Name of the mesh
	unsigned short m_usNumTris;//Number of triangles in the group
	unsigned short * m_uspIndices; //Triangle indices
	char m_cMaterial;          //Material index, -1 = no material

	//Let itclean up after itself like usual
	SMs3dMesh()
	{
		m_uspIndices = 0;
	}
	~SMs3dMesh()
	{
		if(m_uspIndices)
		{
			delete [] m_uspIndices;
			m_uspIndices = 0;
		}
	}


} PACK_STRUCT;

//	Mesh
struct Mesh
{
	int m_materialIndex;
	int m_usNumTris;
	int *m_uspIndices;
};

//	Material properties
struct Material
{
	float m_ambient[4], m_diffuse[4], m_specular[4], m_emissive[4];
	float m_shininess;
	unsigned int m_texture;
	char *m_pTextureFilename;
};

class Ms3dVertexArrayMesh 
{
public:
	Ms3dVertexArrayMesh()
	{
		materialID = 0;
		pVertexArray = NULL;
		numOfVertex = 0;
	}

	~Ms3dVertexArrayMesh()
	{
		if (pVertexArray != NULL)
		{
			delete pVertexArray;
			pVertexArray = NULL;
		}

		numOfVertex = 0;
	}

	int materialID;

	float * pVertexArray;

	int numOfVertex;
};

class Ms3dIntervelData 
{
public:
	Ms3dIntervelData()
	{
		m_pMesh = NULL;
		m_numberOfMesh = 0;
	}

	~Ms3dIntervelData()
	{
		if (m_pMesh != NULL)
		{
			delete[] m_pMesh;
			m_pMesh = NULL;
		}
	}

public:
	Ms3dVertexArrayMesh* m_pMesh;
	int m_numberOfMesh;
	Ms3dVertexArrayMesh* m_pMeshOriginal;
};

#pragma pack(push, packing)
#pragma pack(4)
struct Ms3dVertexArrayElement
{
	float m_fTexcoords[2];
	float m_fBone;
	float m_vNormals[3];
	float m_vVert[3];
};
#pragma pack(pop, packing)


// Default alignment
#ifdef _MSC_VER
#	pragma pack( pop, packing )
#endif

#undef PACK_STRUCT
