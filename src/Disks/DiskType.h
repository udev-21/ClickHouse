#pragma once

#include <common/types.h>

namespace DB
{

enum class DiskType
{
    Local,
    RAM,
    S3,
    HDFS,
    Encrypted
};

inline String toString(DiskType disk_type)
{
    switch (disk_type)
    {
        case DiskType::Local:
            return "local";
        case DiskType::RAM:
            return "memory";
        case DiskType::S3:
            return "s3";
        case DiskType::HDFS:
            return "hdfs";
        case DiskType::Encrypted:
            return "encrypted";
    }
    __builtin_unreachable();
}

}
