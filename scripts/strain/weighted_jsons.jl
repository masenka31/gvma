using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_strain.jl"))

"""
    jaccard_distance(d1::Vector, d2::Vector)

Calculate the Jaccard distance between two vectors
as number of points in intersetion divided by number
of points in the union: d = # intersection / # union.
"""
function jaccard_distance(d1::Vector, d2::Vector)
    #dif = length(setdiff(d1,d2))
    int = length(intersect(d1,d2))
    un = length(union(d1,d2))
    return (un - int)/un
end

# load the Jaccard matrix
using BSON
using LinearAlgebra
L = BSON.load(datadir("jaccard_matrix_strain.bson"))[:L]
# create full matrix
L_full = Symmetric(L)

using ClusterLosses

js1 = JSON.parse("""{ "a" : {"b" : "v", "c" : "u"}}""")
JSON.parse(
    """
    {
    "behavior_summary": {
        "read_files": [
            "C:\\Documents and Settings\\Administrator\\My Documents\\vpnwym.exe",
            "C:\\WINDOWS\\Temp\\hromi.exe"
        ],
        "files": [
            "C:\\WINDOWS\\system32\\user32.dll",
            "C:\\WINDOWS\\system32\\WININET.dll",
            "C:\\WINDOWS\\system32\\gdi32.dll",
            "C:\\WINDOWS\\system32\\advapi32.dll",
            "C:\\WINDOWS\\system32\\kernel32.dll",
            "C:\\Documents and Settings\\Administrator\\My Documents\\vpnwym.exe",
            "C:\\WINDOWS\\system32\\shell32.dll",
            "C:\\WINDOWS\\system32\\ws2_32.dll",
            "C:\\WINDOWS\\system32\\ntdll.dll",
            "C:\\WINDOWS\\Temp\\hromi.exe"
        ],
        "write_files": [
            "C:\\WINDOWS\\Temp\\hromi.exe"
        ],
        "executed_commands": [
            "C:\\WINDOWS\\Temp\\hromi.exe"
        ]
    },
    "network_http": [],
    "info": {
        "source": "gvma"
    },
    "signatures": [
        {
            "name": "heur:read_self",
            "severity": 93,
            "description": "heur:Performs file read operation on itself."
        },
        {
            "name": "heur:dropped_file",
            "severity": 0,
            "description": "heur:Drops a file."
        },
        {
            "name": "heur:dropped_mzpe",
            "severity": 0,
            "description": "heur:Drops a PE file."
        },
        {
            "name": "heur:dropped_exe",
            "severity": 113,
            "description": "heur:Drops a PE file with .EXE file extension."
        },
        {
            "name": "heur:exec_dropped",
            "severity": 188,
            "description": "heur:Executes dropped PE file."
        },
        {
            "name": "heur:shell_execute",
            "severity": 0,
            "description": ""
        }
    ]
}
    """
)