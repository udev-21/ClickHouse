#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <Disks/ObjectStorages/S3/S3ObjectStorage.h>
#include "Common/Exception.h"
#include "Common/ObjectStorageKey.h"
#include "Common/ThreadPool.h"
#include "Interpreters/Cache/QueryCache.h"

#if USE_AWS_S3

#include <IO/S3Common.h>
#include <Disks/ObjectStorages/ObjectStorageIteratorAsync.h>

#include <Disks/IO/ReadBufferFromRemoteFSGather.h>
#include <Disks/IO/AsynchronousBoundedReadBuffer.h>
#include <Disks/IO/ThreadPoolRemoteFSReader.h>
#include <IO/WriteBufferFromS3.h>
#include <IO/ReadBufferFromS3.h>
#include <IO/S3/getObjectInfo.h>
#include <IO/S3/copyS3File.h>
#include <Interpreters/Context.h>
#include <Common/threadPoolCallbackRunner.h>
#include <Core/Settings.h>
#include <IO/S3/BlobStorageLogWriter.h>

#include <Disks/ObjectStorages/S3/diskSettings.h>

#include <Common/ProfileEvents.h>
#include <Common/StringUtils.h>
#include <Common/logger_useful.h>
#include <Common/MultiVersion.h>
#include <Common/Macros.h>

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>


namespace ProfileEvents
{
    extern const Event S3DeleteObjects;
    extern const Event S3ListObjects;
    extern const Event DiskS3DeleteObjects;
    extern const Event DiskS3ListObjects;
}

namespace CurrentMetrics
{
    extern const Metric ObjectStorageS3Threads;
    extern const Metric ObjectStorageS3ThreadsActive;
    extern const Metric ObjectStorageS3ThreadsScheduled;
}


namespace DB
{

namespace ErrorCodes
{
    extern const int S3_ERROR;
    extern const int BAD_ARGUMENTS;
    extern const int LOGICAL_ERROR;
}

namespace HexArithmetics
{

static constexpr std::string_view alphabet = "0123456789abcdefghijklmnopqrstuvwxyz./-";//ABCDEFGHIJKLMNOPQRSTUVWXYZ

class HexFile
{
public:
    explicit HexFile(int x)
        : number_representation(x)
    {
    }

    explicit HexFile(const std::string& filename)
    {
        number_representation = 0;
        for (auto elem : filename) 
        {
            char converted_number;
            auto it = alphabet.find(elem);
            if (it != std::string::npos)
            {
                converted_number = it;
            }
            else 
            {
                std::string error_msg = "S3 doesn't support symbol ";
                error_msg.push_back(elem);
                throw std::runtime_error(error_msg);
                //throw Exception(ErrorCodes::BAD_ARGUMENTS,  "S3 doesn't support symbol", elem);
            }
            number_representation = number_representation * alphabet.size() + converted_number;
        }
    }

    HexFile& operator*=(float value) 
    {
        boost::multiprecision::cpp_dec_float_100 float_representation(number_representation);
        float_representation *= value;
        number_representation = static_cast<boost::multiprecision::cpp_int>(float_representation);
        return *this;
    }

    HexFile operator-(const HexFile& other)
    {
        return HexFile(number_representation - other.number_representation);
    }

    HexFile operator+(const HexFile& other)
    {
        return HexFile(number_representation + other.number_representation);
    }

    HexFile operator*(size_t number)
    {
        return HexFile(number_representation * number);
    }

    bool operator<=(const HexFile& other) const
    {
        return number_representation <= other.number_representation;
    }

    std::string toString() const
    {
        auto current = number_representation;
        std::string answer;
        while (current > 0)
        {
            int token = static_cast<int>(current % alphabet.size());
            if (token < 0 || token >= static_cast<int>(alphabet.size()))
            {
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Error decoding filename");   
            }
            answer.push_back(alphabet[token]);
            current /= alphabet.size();
        }
        std::reverse(answer.begin(), answer.end());
        return answer;
    }

private:
    explicit HexFile(boost::multiprecision::cpp_int number_representation_)
        : number_representation(number_representation_)
    {
    }

    boost::multiprecision::cpp_int number_representation;
};

}

namespace
{

template <typename Result, typename Error>
void throwIfError(const Aws::Utils::Outcome<Result, Error> & response)
{
    if (!response.IsSuccess())
    {
        const auto & err = response.GetError();
        throw S3Exception(
            fmt::format("{} (Code: {}, s3 exception: {})",
                        err.GetMessage(), static_cast<size_t>(err.GetErrorType()), err.GetExceptionName()),
            err.GetErrorType());
    }
}

template <typename Result, typename Error>
void throwIfUnexpectedError(const Aws::Utils::Outcome<Result, Error> & response, bool if_exists)
{
    /// In this case even if absence of key may be ok for us,
    /// the log will be polluted with error messages from aws sdk.
    /// Looks like there is no way to suppress them.

    if (!response.IsSuccess() && (!if_exists || !S3::isNotFoundError(response.GetError().GetErrorType())))
    {
        const auto & err = response.GetError();
        throw S3Exception(err.GetErrorType(), "{} (Code: {})", err.GetMessage(), static_cast<size_t>(err.GetErrorType()));
    }
}

template <typename Result, typename Error>
void logIfError(const Aws::Utils::Outcome<Result, Error> & response, std::function<String()> && msg)
{
    try
    {
        throwIfError(response);
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__, msg());
    }
}

}

namespace
{

class S3IteratorAsync final : public IObjectStorageIteratorAsync
{
public:
    S3IteratorAsync(
        const std::string & bucket_,
        const std::string & path_prefix_,
        std::shared_ptr<const S3::Client> client_,
        size_t max_list_size_)
        : IObjectStorageIteratorAsync(
            CurrentMetrics::ObjectStorageS3Threads,
            CurrentMetrics::ObjectStorageS3ThreadsActive,
            CurrentMetrics::ObjectStorageS3ThreadsScheduled,
            "ListObjectS3")
        , client(client_)
        , request(std::make_unique<S3::ListObjectsV2Request>())
        , bucket(bucket_)
        , path_prefix(path_prefix_)
        , max_list_size(max_list_size_)
    {
        request->SetBucket(bucket_);
        request->SetPrefix(path_prefix_);
        request->SetMaxKeys(static_cast<int>(max_list_size_));
    }

    ~S3IteratorAsync() override
    {
        /// Deactivate background threads before resetting the request to avoid data race.
        deactivate();
        request.reset();
        client.reset();
    }

private:
    class QueryCache 
    {
    public:
        std::vector<Aws::S3::Model::Object> getBatchFrom(const std::string& filename_after, size_t count) const
        {
            std::shared_lock<std::shared_mutex> lock(mutex);

            std::vector<Aws::S3::Model::Object> result;
            result.reserve(count);
            for (auto it = std::upper_bound(extracted_keys.begin(), extracted_keys.end(), filename_after); it != extracted_keys.end(); ++it) {
                result.push_back(key_to_object.at(*it));
                if (result.size() == count) {
                    break;
                }
            }
            return result;
        }

        void insertObjects(const std::vector<Aws::S3::Model::Object>& objects) 
        {
            std::unique_lock<std::shared_mutex> lock(mutex);
                
            for (const auto& object : objects)
            {
                extracted_keys.push_back(object.GetKey());
                key_to_object[object.GetKey()] = object;
            }
        }

        void build()
        {
            std::sort(extracted_keys.begin(), extracted_keys.end());
            extracted_keys.erase(std::unique(extracted_keys.begin(), extracted_keys.end()), extracted_keys.end());
        }

        void clear()
        {
            extracted_keys.clear();
            key_to_object.clear();
        }

    private:
        std::vector<std::string> extracted_keys;
        std::unordered_map<std::string, Aws::S3::Model::Object> key_to_object;

        mutable std::shared_mutex mutex;
    };

    void RunSubrequest(
        const HexArithmetics::HexFile& start_file, 
        const HexArithmetics::HexFile& end_file)
    {
        S3::ListObjectsV2Request current_request;
        current_request.SetBucket(bucket);
        current_request.SetPrefix(path_prefix);
        current_request.SetMaxKeys(static_cast<int>(max_list_size));
        
        bool done = false;
        std::string file_iterator = start_file.toString();

        while (!done)
        {
            current_request.SetStartAfter(file_iterator);
            auto outcome = client->ListObjectsV2(current_request);
            if (outcome.IsSuccess()) {
                const auto& result = outcome.GetResult();
                cache.insertObjects(result.GetContents());
                if (!result.GetIsTruncated() || end_file <= HexArithmetics::HexFile(result.GetContents().back().GetKey()))
                {
                    done = true;
                }
            } else {
                throw S3Exception(outcome.GetError().GetErrorType(),
                    "Could not list objects in bucket {} with prefix {}, S3 exception: {}, message: {}",
                    quoteString(current_request.GetBucket()), quoteString(current_request.GetPrefix()),
                    backQuote(outcome.GetError().GetExceptionName()), quoteString(outcome.GetError().GetMessage()));
            }
        }
    }

    void fillCache(
        const std::string& first_request_file_start,
        const std::string& first_request_file_end,
        size_t num_requests,
        size_t num_worker,
        float lenght_decrease)
    {
        auto file_start = HexArithmetics::HexFile(first_request_file_start);
        auto file_end = HexArithmetics::HexFile(first_request_file_end);
        auto distance = file_end - file_start;
        distance *= lenght_decrease;

        ThreadPool pool_requests(
            CurrentMetrics::ObjectStorageS3Threads,
            CurrentMetrics::ObjectStorageS3ThreadsActive,
            CurrentMetrics::ObjectStorageS3ThreadsScheduled,
            num_worker,
            num_worker,
            0);

        for (size_t i = 0; i < num_requests; ++i)
        {
            auto start_file = file_end + HexArithmetics::HexFile(1) + distance * i;
            auto end_file = file_end + HexArithmetics::HexFile(1) + distance * (i + 1);
            //std::cerr << start_file.toString() << ' ' << end_file.toString() << '\n';
            throw_message += start_file.toString() + "\n";
            throw_message += end_file.toString() + "\n\n";
            pool_requests.scheduleOrThrow([this, start_file, end_file] {
                this->RunSubrequest(start_file, end_file);
            });
        }
        pool_requests.wait();
        cache.build();
    }

    bool getBatchAndCheckNext(RelativePathsWithMetadata & batch) override
    {
        cnt++;
        ProfileEvents::increment(ProfileEvents::S3ListObjects);
        ProfileEvents::increment(ProfileEvents::DiskS3ListObjects);

        auto cache_result = cache.getBatchFrom(request->GetStartAfter(), max_list_size);

        if (cache_result.size() == max_list_size)
        {
            throw_message += std::to_string(cnt) + "\n";
            throw_message += (cache_result[0].GetKey()) + "\n";
            throw_message += (cache_result.back().GetKey()) + "\n\n";

            for (const auto & object : cache_result)
            {
                ObjectMetadata metadata{static_cast<uint64_t>(object.GetSize()), Poco::Timestamp::fromEpochTime(object.GetLastModified().Seconds()), {}};
                batch.emplace_back(std::make_shared<RelativePathWithMetadata>(object.GetKey(), std::move(metadata)));
            }

            if (cnt > 600)
            {
                throw std::runtime_error(throw_message);
            }

            request->SetStartAfter(cache_result.back().GetKey());
            return true;
        }

        auto outcome = client->ListObjectsV2(*request);

        /// Outcome failure will be handled on the caller side.
        if (outcome.IsSuccess())
        {
            auto objects = outcome.GetResult().GetContents();
            for (const auto & object : objects)
            {
                ObjectMetadata metadata{static_cast<uint64_t>(object.GetSize()), Poco::Timestamp::fromEpochTime(object.GetLastModified().Seconds()), {}};
                batch.emplace_back(std::make_shared<RelativePathWithMetadata>(object.GetKey(), std::move(metadata)));
            }
            request->SetStartAfter(objects.back().GetKey());
            if (first_request) {
                cache.clear();
                fillCache(objects[0].GetKey(), objects.back().GetKey(), 900, 100, 1.0f);
                first_request = false;
            } else {
                throw_message += "EXCEEDED AT " + std::to_string(cnt) + "\n";
                throw std::runtime_error(throw_message);
            }
            /// It returns false when all objects were returned
            return outcome.GetResult().GetIsTruncated();
        }

        throw S3Exception(outcome.GetError().GetErrorType(),
                          "Could not list objects in bucket {} with prefix {}, S3 exception: {}, message: {}",
                          quoteString(request->GetBucket()), quoteString(request->GetPrefix()),
                          backQuote(outcome.GetError().GetExceptionName()), quoteString(outcome.GetError().GetMessage()));
    }

    int cnt = 0;
    std::string throw_message = "";

    bool first_request = true;
    std::shared_ptr<const S3::Client> client;
    std::unique_ptr<S3::ListObjectsV2Request> request;
    QueryCache cache;
    std::string bucket;
    std::string path_prefix;
    size_t max_list_size;
};

}

bool S3ObjectStorage::exists(const StoredObject & object) const
{
    auto settings_ptr = s3_settings.get();
    return S3::objectExists(*client.get(), uri.bucket, object.remote_path, {});
}

std::unique_ptr<ReadBufferFromFileBase> S3ObjectStorage::readObjects( /// NOLINT
    const StoredObjects & objects,
    const ReadSettings & read_settings,
    std::optional<size_t>,
    std::optional<size_t>) const
{
    ReadSettings disk_read_settings = patchSettings(read_settings);
    auto global_context = Context::getGlobalContextInstance();

    auto settings_ptr = s3_settings.get();

    auto read_buffer_creator =
        [this, settings_ptr, disk_read_settings]
        (bool restricted_seek, const StoredObject & object_) -> std::unique_ptr<ReadBufferFromFileBase>
    {
        return std::make_unique<ReadBufferFromS3>(
            client.get(),
            uri.bucket,
            object_.remote_path,
            uri.version_id,
            settings_ptr->request_settings,
            disk_read_settings,
            /* use_external_buffer */true,
            /* offset */0,
            /* read_until_position */0,
            restricted_seek);
    };

    switch (read_settings.remote_fs_method)
    {
        case RemoteFSReadMethod::read:
        {
            return std::make_unique<ReadBufferFromRemoteFSGather>(
                std::move(read_buffer_creator),
                objects,
                "s3:" + uri.bucket + "/",
                disk_read_settings,
                global_context->getFilesystemCacheLog(),
                /* use_external_buffer */false);
        }
        case RemoteFSReadMethod::threadpool:
        {
            auto impl = std::make_unique<ReadBufferFromRemoteFSGather>(
                std::move(read_buffer_creator),
                objects,
                "s3:" + uri.bucket + "/",
                disk_read_settings,
                global_context->getFilesystemCacheLog(),
                /* use_external_buffer */true);

            auto & reader = global_context->getThreadPoolReader(FilesystemReaderType::ASYNCHRONOUS_REMOTE_FS_READER);
            return std::make_unique<AsynchronousBoundedReadBuffer>(
                std::move(impl), reader, disk_read_settings,
                global_context->getAsyncReadCounters(),
                global_context->getFilesystemReadPrefetchesLog());
        }
    }
}

std::unique_ptr<ReadBufferFromFileBase> S3ObjectStorage::readObject( /// NOLINT
    const StoredObject & object,
    const ReadSettings & read_settings,
    std::optional<size_t>,
    std::optional<size_t>) const
{
    auto settings_ptr = s3_settings.get();
    return std::make_unique<ReadBufferFromS3>(
        client.get(),
        uri.bucket,
        object.remote_path,
        uri.version_id,
        settings_ptr->request_settings,
        patchSettings(read_settings));
}

std::unique_ptr<WriteBufferFromFileBase> S3ObjectStorage::writeObject( /// NOLINT
    const StoredObject & object,
    WriteMode mode, // S3 doesn't support append, only rewrite
    std::optional<ObjectAttributes> attributes,
    size_t buf_size,
    const WriteSettings & write_settings)
{
    WriteSettings disk_write_settings = IObjectStorage::patchSettings(write_settings);

    if (mode != WriteMode::Rewrite)
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "S3 doesn't support append to files");

    S3::RequestSettings request_settings = s3_settings.get()->request_settings;
    /// NOTE: For background operations settings are not propagated from session or query. They are taken from
    /// default user's .xml config. It's obscure and unclear behavior. For them it's always better
    /// to rely on settings from disk.
    if (auto query_context = CurrentThread::getQueryContext();
        query_context && !query_context->isBackgroundOperationContext())
    {
        const auto & settings = query_context->getSettingsRef();
        request_settings.updateFromSettings(settings, /* if_changed */true, settings.s3_validate_request_settings);
    }

    ThreadPoolCallbackRunnerUnsafe<void> scheduler;
    if (write_settings.s3_allow_parallel_part_upload)
        scheduler = threadPoolCallbackRunnerUnsafe<void>(getThreadPoolWriter(), "VFSWrite");

    auto blob_storage_log = BlobStorageLogWriter::create(disk_name);
    if (blob_storage_log)
        blob_storage_log->local_path = object.local_path;

    return std::make_unique<WriteBufferFromS3>(
        client.get(),
        uri.bucket,
        object.remote_path,
        buf_size,
        request_settings,
        std::move(blob_storage_log),
        attributes,
        std::move(scheduler),
        disk_write_settings);
}


ObjectStorageIteratorPtr S3ObjectStorage::iterate(const std::string & path_prefix, size_t max_keys) const
{
    auto settings_ptr = s3_settings.get();
    if (!max_keys)
        max_keys = settings_ptr->list_object_keys_size;
    return std::make_shared<S3IteratorAsync>(uri.bucket, path_prefix, client.get(), max_keys);
}

void S3ObjectStorage::listObjects(const std::string & path, RelativePathsWithMetadata & children, size_t max_keys) const
{
    auto settings_ptr = s3_settings.get();

    S3::ListObjectsV2Request request;
    request.SetBucket(uri.bucket);
    request.SetPrefix(path);
    if (max_keys)
        request.SetMaxKeys(static_cast<int>(max_keys));
    else
        request.SetMaxKeys(settings_ptr->list_object_keys_size);

    Aws::S3::Model::ListObjectsV2Outcome outcome;
    do
    {
        ProfileEvents::increment(ProfileEvents::S3ListObjects);
        ProfileEvents::increment(ProfileEvents::DiskS3ListObjects);

        outcome = client.get()->ListObjectsV2(request);
        throwIfError(outcome);

        auto result = outcome.GetResult();
        auto objects = result.GetContents();

        if (objects.empty())
            break;

        for (const auto & object : objects)
            children.emplace_back(std::make_shared<RelativePathWithMetadata>(
                object.GetKey(),
                ObjectMetadata{
                    static_cast<uint64_t>(object.GetSize()),
                    Poco::Timestamp::fromEpochTime(object.GetLastModified().Seconds()),
                    {}}));

        if (max_keys)
        {
            size_t keys_left = max_keys - children.size();
            if (keys_left <= 0)
                break;
            request.SetMaxKeys(static_cast<int>(keys_left));
        }

        request.SetContinuationToken(outcome.GetResult().GetNextContinuationToken());
    } while (outcome.GetResult().GetIsTruncated());
}

void S3ObjectStorage::removeObjectImpl(const StoredObject & object, bool if_exists)
{
    ProfileEvents::increment(ProfileEvents::S3DeleteObjects);
    ProfileEvents::increment(ProfileEvents::DiskS3DeleteObjects);

    S3::DeleteObjectRequest request;
    request.SetBucket(uri.bucket);
    request.SetKey(object.remote_path);
    auto outcome = client.get()->DeleteObject(request);
    if (auto blob_storage_log = BlobStorageLogWriter::create(disk_name))
        blob_storage_log->addEvent(BlobStorageLogElement::EventType::Delete,
                                   uri.bucket, object.remote_path, object.local_path, object.bytes_size,
                                   outcome.IsSuccess() ? nullptr : &outcome.GetError());

    throwIfUnexpectedError(outcome, if_exists);

    LOG_DEBUG(log, "Object with path {} was removed from S3", object.remote_path);
}

void S3ObjectStorage::removeObjectsImpl(const StoredObjects & objects, bool if_exists)
{
    if (objects.empty())
        return;

    if (!s3_capabilities.support_batch_delete)
    {
        for (const auto & object : objects)
            removeObjectImpl(object, if_exists);
    }
    else
    {
        auto settings_ptr = s3_settings.get();

        size_t chunk_size_limit = settings_ptr->objects_chunk_size_to_delete;
        size_t current_position = 0;

        auto blob_storage_log = BlobStorageLogWriter::create(disk_name);
        while (current_position < objects.size())
        {
            std::vector<Aws::S3::Model::ObjectIdentifier> current_chunk;
            String keys;
            size_t first_position = current_position;
            for (; current_position < objects.size() && current_chunk.size() < chunk_size_limit; ++current_position)
            {
                Aws::S3::Model::ObjectIdentifier obj;
                obj.SetKey(objects[current_position].remote_path);
                current_chunk.push_back(obj);

                if (!keys.empty())
                    keys += ", ";
                keys += objects[current_position].remote_path;
            }

            Aws::S3::Model::Delete delkeys;
            delkeys.SetObjects(current_chunk);

            ProfileEvents::increment(ProfileEvents::S3DeleteObjects);
            ProfileEvents::increment(ProfileEvents::DiskS3DeleteObjects);
            S3::DeleteObjectsRequest request;
            request.SetBucket(uri.bucket);
            request.SetDelete(delkeys);
            auto outcome = client.get()->DeleteObjects(request);

            if (blob_storage_log)
            {
                const auto * outcome_error = outcome.IsSuccess() ? nullptr : &outcome.GetError();
                auto time_now = std::chrono::system_clock::now();
                for (size_t i = first_position; i < current_position; ++i)
                    blob_storage_log->addEvent(BlobStorageLogElement::EventType::Delete,
                                               uri.bucket, objects[i].remote_path, objects[i].local_path, objects[i].bytes_size,
                                               outcome_error, time_now);
            }

            LOG_DEBUG(log, "Objects with paths [{}] were removed from S3", keys);
            throwIfUnexpectedError(outcome, if_exists);
        }
    }
}

void S3ObjectStorage::removeObject(const StoredObject & object)
{
    removeObjectImpl(object, false);
}

void S3ObjectStorage::removeObjectIfExists(const StoredObject & object)
{
    removeObjectImpl(object, true);
}

void S3ObjectStorage::removeObjects(const StoredObjects & objects)
{
    removeObjectsImpl(objects, false);
}

void S3ObjectStorage::removeObjectsIfExist(const StoredObjects & objects)
{
    removeObjectsImpl(objects, true);
}

std::optional<ObjectMetadata> S3ObjectStorage::tryGetObjectMetadata(const std::string & path) const
{
    auto settings_ptr = s3_settings.get();
    auto object_info = S3::getObjectInfo(
        *client.get(), uri.bucket, path, {}, /* with_metadata= */ true, /* throw_on_error= */ false);

    if (object_info.size == 0 && object_info.last_modification_time == 0 && object_info.metadata.empty())
        return {};

    ObjectMetadata result;
    result.size_bytes = object_info.size;
    result.last_modified = Poco::Timestamp::fromEpochTime(object_info.last_modification_time);
    result.attributes = object_info.metadata;

    return result;
}

ObjectMetadata S3ObjectStorage::getObjectMetadata(const std::string & path) const
{
    auto settings_ptr = s3_settings.get();
    S3::ObjectInfo object_info;
    try
    {
        object_info = S3::getObjectInfo(*client.get(), uri.bucket, path, {}, /* with_metadata= */ true);
    }
    catch (DB::Exception & e)
    {
        e.addMessage("while reading " + path);
        throw;
    }

    ObjectMetadata result;
    result.size_bytes = object_info.size;
    result.last_modified = Poco::Timestamp::fromEpochTime(object_info.last_modification_time);
    result.attributes = object_info.metadata;

    return result;
}

void S3ObjectStorage::copyObjectToAnotherObjectStorage( // NOLINT
    const StoredObject & object_from,
    const StoredObject & object_to,
    const ReadSettings & read_settings,
    const WriteSettings & write_settings,
    IObjectStorage & object_storage_to,
    std::optional<ObjectAttributes> object_to_attributes)
{
    /// Shortcut for S3
    if (auto * dest_s3 = dynamic_cast<S3ObjectStorage * >(&object_storage_to); dest_s3 != nullptr)
    {
        auto current_client = dest_s3->client.get();
        auto settings_ptr = s3_settings.get();
        auto size = S3::getObjectSize(*current_client, uri.bucket, object_from.remote_path, {});
        auto scheduler = threadPoolCallbackRunnerUnsafe<void>(getThreadPoolWriter(), "S3ObjStor_copy");

        try
        {
            copyS3File(
                /*src_s3_client=*/current_client,
                /*src_bucket=*/uri.bucket,
                /*src_key=*/object_from.remote_path,
                /*src_offset=*/0,
                /*src_size=*/size,
                /*dest_s3_client=*/current_client,
                /*dest_bucket=*/dest_s3->uri.bucket,
                /*dest_key=*/object_to.remote_path,
                settings_ptr->request_settings,
                patchSettings(read_settings),
                BlobStorageLogWriter::create(disk_name),
                object_to_attributes,
                scheduler);
            return;
        }
        catch (S3Exception & exc)
        {
            /// If authentication/permissions error occurs then fallthrough to copy with buffer.
            if (exc.getS3ErrorCode() != Aws::S3::S3Errors::ACCESS_DENIED)
                throw;
            LOG_WARNING(getLogger("S3ObjectStorage"),
                "S3-server-side copy object from the disk {} to the disk {} can not be performed: {}\n",
                getName(), dest_s3->getName(), exc.what());
        }
    }

    IObjectStorage::copyObjectToAnotherObjectStorage(object_from, object_to, read_settings, write_settings, object_storage_to, object_to_attributes);
}

void S3ObjectStorage::copyObject( // NOLINT
    const StoredObject & object_from,
    const StoredObject & object_to,
    const ReadSettings & read_settings,
    const WriteSettings &,
    std::optional<ObjectAttributes> object_to_attributes)
{
    auto current_client = client.get();
    auto settings_ptr = s3_settings.get();
    auto size = S3::getObjectSize(*current_client, uri.bucket, object_from.remote_path, {});
    auto scheduler = threadPoolCallbackRunnerUnsafe<void>(getThreadPoolWriter(), "S3ObjStor_copy");

    copyS3File(
        /*src_s3_client=*/current_client,
        /*src_bucket=*/uri.bucket,
        /*src_key=*/object_from.remote_path,
        /*src_offset=*/0,
        /*src_size=*/size,
        /*dest_s3_client=*/current_client,
        /*dest_bucket=*/uri.bucket,
        /*dest_key=*/object_to.remote_path,
        settings_ptr->request_settings,
        patchSettings(read_settings),
        BlobStorageLogWriter::create(disk_name),
        object_to_attributes,
        scheduler);
}

void S3ObjectStorage::setNewSettings(std::unique_ptr<S3ObjectStorageSettings> && s3_settings_)
{
    s3_settings.set(std::move(s3_settings_));
}

void S3ObjectStorage::shutdown()
{
    /// This call stops any next retry attempts for ongoing S3 requests.
    /// If S3 request is failed and the method below is executed S3 client immediately returns the last failed S3 request outcome.
    /// If S3 is healthy nothing wrong will be happened and S3 requests will be processed in a regular way without errors.
    /// This should significantly speed up shutdown process if S3 is unhealthy.
    const_cast<S3::Client &>(*client.get()).DisableRequestProcessing();
}

void S3ObjectStorage::startup()
{
    /// Need to be enabled if it was disabled during shutdown() call.
    const_cast<S3::Client &>(*client.get()).EnableRequestProcessing();
}

void S3ObjectStorage::applyNewSettings(
    const Poco::Util::AbstractConfiguration & config,
    const std::string & config_prefix,
    ContextPtr context,
    const ApplyNewSettingsOptions & options)
{
    auto settings_from_config = getSettings(config, config_prefix, context, uri.uri_str, context->getSettingsRef().s3_validate_request_settings);
    auto modified_settings = std::make_unique<S3ObjectStorageSettings>(*s3_settings.get());
    modified_settings->auth_settings.updateIfChanged(settings_from_config->auth_settings);
    modified_settings->request_settings.updateIfChanged(settings_from_config->request_settings);

    if (auto endpoint_settings = context->getStorageS3Settings().getSettings(uri.uri.toString(), context->getUserName()))
    {
        modified_settings->auth_settings.updateIfChanged(endpoint_settings->auth_settings);
        modified_settings->request_settings.updateIfChanged(endpoint_settings->request_settings);
    }

    auto current_settings = s3_settings.get();
    if (options.allow_client_change
        && (current_settings->auth_settings.hasUpdates(modified_settings->auth_settings) || for_disk_s3))
    {
        auto new_client = getClient(uri, *modified_settings, context, for_disk_s3);
        client.set(std::move(new_client));
    }
    s3_settings.set(std::move(modified_settings));
}

std::unique_ptr<IObjectStorage> S3ObjectStorage::cloneObjectStorage(
    const std::string & new_namespace,
    const Poco::Util::AbstractConfiguration & config,
    const std::string & config_prefix,
    ContextPtr context)
{
    const auto & settings = context->getSettingsRef();
    auto new_s3_settings = getSettings(config, config_prefix, context, uri.uri_str, settings.s3_validate_request_settings);
    auto new_client = getClient(uri, *new_s3_settings, context, for_disk_s3);

    auto new_uri{uri};
    new_uri.bucket = new_namespace;

    return std::make_unique<S3ObjectStorage>(
        std::move(new_client), std::move(new_s3_settings), new_uri, s3_capabilities, key_generator, disk_name);
}

ObjectStorageKey S3ObjectStorage::generateObjectKeyForPath(const std::string & path) const
{
    if (!key_generator)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Key generator is not set");

    return key_generator->generate(path, /* is_directory */ false);
}

std::shared_ptr<const S3::Client> S3ObjectStorage::getS3StorageClient()
{
    return client.get();
}

std::shared_ptr<const S3::Client> S3ObjectStorage::tryGetS3StorageClient()
{
    return client.get();
}
}

#endif
